"""
finetune_subcompartments.py
===========================

Transfer-learning fine-tune of TECSAS to predict chromatin **subcompartment**
annotations (A1 / A2 / B1 / B2 / B3) for a *new* cell type at 50 kb / hg19,
using **your own** Hi-C-derived subcompartment labels for training.

Why transfer learning?
----------------------
TECSAS' ``encoder`` (``nn.Linear(1, d_model)``) and ``transformer_encoder``
operate per sequence position, so they are independent of how many epigenomic
experiments (NEXP) a cell type has -- only the output head ``l2`` depends on
``nfeatures = NEXP * (2 * n_neighbors + 1)``. We therefore reuse the
pre-trained GM12878 encoder as a frozen feature extractor and train only a
fresh ``l2`` head sized to the target cell line's data. This mirrors the
transfer-learning recipe in ``train_and_predict_XADS_HistMod_RNASeq.ipynb``,
but keeps the original 5-class subcompartment task instead of switching to the
2-class nuclear-body task.

What you must provide
---------------------
1. ``cell_line`` -- an ENCODE-recognised name; its histone (and optionally TF)
   ChIP-seq tracks are downloaded and processed automatically.
2. ``labels_dir`` -- a directory with YOUR ground-truth subcompartment labels,
   one file per chromosome named ``chr{1..22}_beads.txt.original``:
       * space-delimited, two columns: ``<bin_index> <label>``
       * label in column index 1, one of ``A1 A2 B1 B2 B3`` ( ``NA`` allowed
         and is excluded from training; ``B4`` is also excluded)
       * one row per 50 kb bin, in order, with the bin count for each
         chromosome matching ``data_process.chrm_size`` (hg19 @ 50 kb).
   See ``TECSAS/share/subcom_GM12878_50kb/`` for a concrete example of the
   format (those are the shipped GM12878 labels).

Outputs (written to ``output_dir``)
-----------------------------------
* ``best_val_model_params_<cell_line>.pt`` -- best fine-tuned weights
* ``subcompartments_<cell_line>_predictions.txt`` -- integer labels
  (0=A1, 1=A2, 2=B1, 3=B2, 4=B3)
* ``subcompartments_<cell_line>.bed`` -- colored BED track for genome browsers
* ``vloss_<cell_line>.png`` -- validation-loss curve

Usage
-----
As a script::

    python Tutorials/finetune_subcompartments.py \
        --cell-line MyCell --labels-dir /path/to/my_labels_50kb \
        --pretrained TECSAS/share/models/bv_GM12878_155.pt \
        --output-dir ./finetune_MyCell --epochs 75 --nproc 10

Or edit the ``Config`` defaults below and run without arguments. From a
notebook, import :func:`run_finetune` and pass a :class:`Config`.

Note: a real run needs network access (ENCODE), your label files, and ideally a
GPU. The heavy lifting reuses existing ``data_process`` methods in
``TECSAS/TECSAS.py`` -- nothing here re-implements data download or track
processing.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# ``TECSAS`` exports ``data_process`` and the ``TECSAS`` model class.
from TECSAS.TECSAS import TECSAS as TECSASModel
from TECSAS.TECSAS import data_process

# Subcompartment label vocabulary (must match data_process.TYPE_TO_INT ordering).
INT_TO_TYPE = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "B3"}
# Genome-browser colors (itemRgb) per predicted class index.
BED_COLORS = {
    0: "245,47,47",   # A1
    1: "145,47,47",   # A2
    2: "47,187,224",  # B1
    3: "47,47,224",   # B2
    4: "47,47,139",   # B3
}


@dataclass
class Config:
    """All knobs for the fine-tune workflow. Defaults match the shipped
    GM12878 155-experiment / 50 kb / hg19 model architecture."""

    # --- Target cell line + labels ---
    cell_line: str = "MyCell"
    assembly: str = "hg19"
    labels_dir: str = "./my_labels_50kb"      # your chr*_beads.txt.original files
    res: int = 50                             # kb resolution (keep 50 for the shipped model)

    # --- Which assays to download from ENCODE ---
    histones: bool = True
    tf: bool = True
    atac: bool = False
    small_rna: bool = False
    total_rna: bool = False

    # --- Model / windowing (must match the pre-trained checkpoint) ---
    n_neighbors: int = 14
    n_predict: int = 3
    emsize: int = 128
    nhead: int = 8
    d_hid: int = 64
    nlayers: int = 2
    dropout: float = 0.01
    ostates: int = 5                          # 5 subcompartment classes

    # --- Data split (by chromosome) ---
    train_chroms: List[int] = field(default_factory=lambda: list(range(2, 23, 2)))  # even
    test_chroms: List[int] = field(default_factory=lambda: list(range(1, 23, 2)))   # odd
    train_per: float = 0.8

    # --- Training ---
    pretrained: str = "TECSAS/share/models/bv_GM12878_155.pt"
    epochs: int = 75
    nbatches: int = 50
    lr: float = 1.0
    weight_decay: float = 1e-4
    patience: int = 5                         # LR-decay patience (epochs w/o improvement)

    # --- I/O / infra ---
    output_dir: str = "./finetune_output"
    nproc: int = 10
    download: bool = True                     # set False to reuse already-processed tracks
    seed: int = 0


# --------------------------------------------------------------------------- #
# Data preparation (reuses data_process from TECSAS/TECSAS.py)
# --------------------------------------------------------------------------- #
def build_data_process(cfg: Config) -> data_process:
    """Instantiate ``data_process`` for the target cell line, pointing
    ``types_path`` at the user's own subcompartment labels."""
    return data_process(
        cell_line=cfg.cell_line,
        assembly=cfg.assembly,
        organism="human",
        signal_type="signal p-value",
        histones=cfg.histones,
        tf=cfg.tf,
        atac=cfg.atac,
        small_rna=cfg.small_rna,
        total_rna=cfg.total_rna,
        types_path=cfg.labels_dir,
        res=cfg.res,
        require_ENCODE=True,
    )


def prepare_data(cfg: Config, dp: data_process):
    """Download/process ENCODE tracks (optional) and assemble train/val/test
    tensors via ``data_process.training_data_chrom``.

    Returns ``(train_data, val_data, test_data, ntest_loci, nfeatures)``.
    """
    if cfg.download:
        print(f"[data] Downloading + processing ENCODE tracks for {cfg.cell_line} ...")
        dp.download_and_process_cell_line_data(nproc=cfg.nproc)
        print("[data] Downloading + processing GM12878 reference tracks ...")
        dp.download_and_process_ref_data(nproc=cfg.nproc)
        # Screen the target's experiments against the reference (writes
        # unique_exp.txt); warns if fewer than ~5 usable experiments.
        try:
            dp.filter_exp()
        except Exception as exc:  # filter_exp is best-effort; keep going if it errors
            print(f"[data] filter_exp() skipped: {exc}")

    # Train/validation from one chromosome set ...
    train_set, val_set, _, _, _ = dp.training_data_chrom(
        train_chroms=cfg.train_chroms,
        n_neigbors=cfg.n_neighbors,
        train_per=cfg.train_per,
        n_predict=cfg.n_predict,
    )
    # ... test from the held-out chromosome set (train_per=0 -> everything is test).
    test_set, _, _, test_averages, _ = dp.training_data_chrom(
        train_chroms=cfg.test_chroms,
        n_neigbors=cfg.n_neighbors,
        train_per=0.0,
        n_predict=cfg.n_predict,
    )

    train_data = torch.tensor(train_set, dtype=torch.float)
    val_data = torch.tensor(val_set, dtype=torch.float)
    test_data = torch.tensor(test_set, dtype=torch.float)

    ntest_loci = np.arange(len(test_averages[0]))
    nfeatures = train_data.size()[1] - (2 * (cfg.n_predict - 1) + 1)
    print(
        f"[data] train={tuple(train_data.size())} val={tuple(val_data.size())} "
        f"test={tuple(test_data.size())} nfeatures={nfeatures}"
    )
    return train_data, val_data, test_data, ntest_loci, nfeatures


# --------------------------------------------------------------------------- #
# Model + transfer learning
# --------------------------------------------------------------------------- #
def build_model(cfg: Config, nfeatures: int, device: torch.device) -> TECSASModel:
    return TECSASModel(
        cfg.n_predict, cfg.emsize, cfg.nhead, cfg.d_hid, cfg.nlayers,
        nfeatures, cfg.ostates, cfg.dropout,
    ).to(device)


def load_pretrained_encoder(model: TECSASModel, weights_path: str,
                            device: torch.device) -> TECSASModel:
    """Load ONLY the encoder + transformer weights from a pre-trained TECSAS
    checkpoint, freeze them, and leave a fresh (trainable) ``l2`` head.

    The head is intentionally skipped because it is NEXP-dependent and will not
    match a target cell line with a different experiment count. Checkpoints
    saved under ``nn.DataParallel`` carry a ``module.`` prefix that is stripped.
    """
    state = torch.load(weights_path, map_location=device)
    stripped = {}
    for k, v in state.items():
        nk = ".".join(k.split(".")[1:]) if k.startswith("module.") else k
        stripped[nk] = v

    frozen_prefixes = ("encoder.", "transformer_encoder.", "pos_encoder.")
    encoder_state = {k: v for k, v in stripped.items() if k.startswith(frozen_prefixes)}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"[model] loaded {len(encoder_state)} encoder tensors "
          f"(skipped head; {len(unexpected)} unexpected keys ignored)")

    # Freeze the transferred feature extractor; train only the new head.
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith(("encoder.", "transformer_encoder."))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable params: {trainable:,} / {total:,} (head only)")
    return model


# --------------------------------------------------------------------------- #
# Train / evaluate / predict (batch layout mirrors the TECSAS tutorials)
# --------------------------------------------------------------------------- #
def _split_batch(source: Tensor, i: int, bptt: int, n_predict: int,
                 device: torch.device) -> Tuple[Tensor, Tensor]:
    """First ``2*(n_predict-1)+1`` columns are the per-locus targets; the rest
    are the flattened per-experiment features fed to the model as a sequence."""
    npred_cols = 2 * (n_predict - 1) + 1
    rows = source[i * bptt:(i + 1) * bptt]
    data = rows[:, npred_cols:][:, :, None]
    target = rows[:, :npred_cols]
    return data.to(device), target.to(device)


def train_one_epoch(model, train_data, optimizer, criterion, cfg, device, bptt):
    model.train()
    total_loss = 0.0
    npred_cols = 2 * (cfg.n_predict - 1) + 1
    order = list(range(cfg.nbatches))
    np.random.shuffle(order)
    for batch in order:
        data, targets = _split_batch(train_data, batch, bptt, cfg.n_predict, device)
        output, _ = model(data, None)
        loss = 0
        for n in range(npred_cols):
            loss += criterion(output[:, n], targets[:, n].long().to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def evaluate(model, eval_data, criterion, cfg, device, bptt):
    model.eval()
    total_loss = 0.0
    npred_cols = 2 * (cfg.n_predict - 1) + 1
    for batch in range(len(eval_data) // bptt):
        data, targets = _split_batch(eval_data, batch, bptt, cfg.n_predict, device)
        output, _ = model(data, None)
        for n in range(npred_cols):
            total_loss += criterion(output[:, n], targets[:, n].long().to(device)).item()
    return total_loss


def fit(model, train_data, val_data, cfg, device, ckpt_path):
    """Fine-tune the head; checkpoint the best-validation weights. Returns the
    per-epoch validation losses."""
    bptt = max(1, len(train_data) // cfg.nbatches)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

    best_val, best_train, stale = float("inf"), float("inf"), 0
    vlosses = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, cfg, device, bptt)
        val_loss = evaluate(model, val_data, criterion, cfg, device, bptt)
        vlosses.append(val_loss)
        print(f"| epoch {epoch:3d} | {time.time() - t0:5.1f}s | "
              f"train {train_loss:7.2f} | val {val_loss:7.2f} | "
              f"lr {scheduler.get_last_lr()[0]:.5f} | best val {best_val:7.2f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        if train_loss < best_train:
            best_train = train_loss
            stale = 0
        else:
            stale += 1
        if stale > cfg.patience:
            scheduler.step()
            stale = 0
    print(f"= fine-tune done | best val loss {best_val:.2f} -> {ckpt_path}")
    return vlosses


@torch.no_grad()
def predict(model, test_data, ntest_loci, cfg, device, bptt):
    """Run inference and return ``(pred, truth, loci)`` for the test set."""
    model.eval()
    preds, truth, loci = [], [], []
    for batch in range(len(test_data) // bptt):
        data, targets = _split_batch(test_data, batch, bptt, cfg.n_predict, device)
        out = model(data, None)[0].argmax(dim=-1)[:, cfg.n_predict - 1].cpu()
        preds.append(out)
        truth.append(targets[:, cfg.n_predict - 1].cpu())
        loci.append(ntest_loci[batch * bptt:(batch + 1) * bptt])
    return (np.concatenate(preds), np.concatenate(truth), np.concatenate(loci))


def write_outputs(dp, cfg, pred, truth, loci, out_dir):
    """Save predictions as a plain-text label file and a colored BED track."""
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"subcompartments_{cfg.cell_line}_predictions.txt")
    np.savetxt(txt_path, pred, fmt="%d",
               header="TECSAS predictions (0=A1, 1=A2, 2=B1, 3=B2, 4=B3)")

    # Map flat loci back to (chromosome, bin) using cumulative chromosome sizes.
    # Loci are shifted by n_neighbors to undo the leading trim from windowing.
    # Bins are 0-based and span [bin*res, (bin+1)*res) -> always non-negative.
    bed_path = os.path.join(out_dir, f"subcompartments_{cfg.cell_line}.bed")
    res_bp = cfg.res * 1000
    chrm_size = np.asarray(dp.chrm_size)
    cum_end = np.cumsum(chrm_size)              # exclusive global end index per chromosome
    chrom_start = np.concatenate([[0], cum_end[:-1]])
    loci_global = loci + cfg.n_neighbors
    with open(bed_path, "w") as fh:
        for i in range(len(loci_global)):
            g = int(loci_global[i])
            cid = min(int(np.searchsorted(cum_end, g, side="right")), len(chrm_size) - 1)
            binpos = max(0, g - int(chrom_start[cid]))
            start, end = binpos * res_bp, (binpos + 1) * res_bp
            label = int(pred[i])
            color = BED_COLORS.get(label, "0,0,0")
            fh.write(f"chr{cid + 1}\t{start}\t{end}\t{INT_TO_TYPE.get(label, label)}\t"
                     f"{label}\t.\t{start}\t{end}\t{color}\n")

    acc = float(np.mean(pred == truth)) if len(pred) else float("nan")
    print(f"[out] accuracy on held-out chromosomes: {acc:.4f}")
    print(f"[out] predictions -> {txt_path}")
    print(f"[out] BED track    -> {bed_path}")
    return txt_path, bed_path, acc


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_finetune(cfg: Config):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device} cell_line={cfg.cell_line} res={cfg.res}kb")

    dp = build_data_process(cfg)
    train_data, val_data, test_data, ntest_loci, nfeatures = prepare_data(cfg, dp)

    model = build_model(cfg, nfeatures, device)
    model = load_pretrained_encoder(model, cfg.pretrained, device)

    ckpt = os.path.join(cfg.output_dir, f"best_val_model_params_{cfg.cell_line}.pt")
    fit(model, train_data, val_data, cfg, device, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    bptt = max(1, len(train_data) // cfg.nbatches)
    pred, truth, loci = predict(model, test_data, ntest_loci, cfg, device, bptt)
    return write_outputs(dp, cfg, pred, truth, loci, cfg.output_dir)


def _parse_args(argv=None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--cell-line", default=cfg.cell_line)
    p.add_argument("--assembly", default=cfg.assembly)
    p.add_argument("--labels-dir", default=cfg.labels_dir)
    p.add_argument("--res", type=int, default=cfg.res)
    p.add_argument("--pretrained", default=cfg.pretrained)
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--nbatches", type=int, default=cfg.nbatches)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--nproc", type=int, default=cfg.nproc)
    p.add_argument("--no-tf", action="store_true", help="disable TF ChIP-seq download")
    p.add_argument("--no-download", action="store_true",
                   help="reuse already-processed tracks instead of downloading")
    a = p.parse_args(argv)
    return Config(
        cell_line=a.cell_line, assembly=a.assembly, labels_dir=a.labels_dir,
        res=a.res, pretrained=a.pretrained, output_dir=a.output_dir,
        epochs=a.epochs, nbatches=a.nbatches, lr=a.lr, nproc=a.nproc,
        tf=not a.no_tf, download=not a.no_download,
    )


if __name__ == "__main__":
    run_finetune(_parse_args())
