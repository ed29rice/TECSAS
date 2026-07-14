# TECSAS

**Transformer of Epigenetics to Chromatin Structural AnnotationS**

[Documentation](#resources) | [Tutorials](Tutorials/) | [Installation](#installation)

## Overview

**TECSAS** (Transformer of Epigenetics to Chromatin Structural AnnotationS) is a deep learning model based on the Transformer architecture designed to predict chromatin subcompartment annotations directly from epigenomic data. TECSAS leverages information from histone modifications, transcription factor binding profiles, and RNA-seq data to decode the relationship between the biochemical composition of chromatin and its 3D structural behavior.

Chromatin within the nucleus adopts complex three-dimensional structures that are crucial for gene regulation and cellular function. Recent studies have revealed the presence of distinct chromatin subcompartments beyond the traditional A/B compartments (eu- and hetero-chromatin), each exhibiting unique structural and functional properties. TECSAS achieves high accuracy in predicting subcompartment annotations and reveals the influence of long-range epigenomic context on chromatin organization.

![TECSAS Overview](images/tecsas_overview.png)

The framework enables:
- **Chromatin subcompartment prediction**: Classification of genomic regions into subcompartments (A1, A2, B1, B2, B3) at 25-50kb resolution
- **Nuclear body association prediction**: Identification of lamina-associated domains (LADs), nucleolus-associated domains (NADs), and nuclear speckle-associated domains (SPADs)
- **Transfer learning**: Pre-trained encoder on reference cell lines (e.g., GM12878) can be fine-tuned for target cell lines

TECSAS processes epigenomic signal tracks at specified genomic resolution (default 50kb bins), normalizes signals using z-score standardization, and uses sliding window context (default ±14 neighboring bins) to capture spatial dependencies. Unlike methods that rely on Hi-C contact maps, TECSAS predicts 3D genome organization directly from the epigenome, enabling analysis across diverse cell types without requiring proximity ligation experiments.

## Usage

For complete examples, see the [Tutorials](Tutorials/) directory.

## Resources

- **Tutorials**: Step-by-step notebooks in the [Tutorials/](Tutorials/) directory
  - `Load_model_GM12878_155exp_50kbp.ipynb`: Load and use pre-trained GM12878 subcompartment model
  - `Test_GM12878_155exp_50kbp.ipynb`: Evaluate the pre-trained GM12878 model with per-class accuracy and confusion matrices
  - `Load_model_K562_124_exp_25kbp.ipynb`: Load K562 model at 25kb resolution
  - `train_and_predict_HistMod_example.ipynb`: Training workflow using histone modifications
  - `train_and_predict_XADS_HistMod_RNASeq.ipynb`: Complete workflow for nuclear body association (LADs/NADs/SPADs) prediction using transfer learning
  - `Finetune_subcompartments_new_cell_50kb.ipynb` / `finetune_subcompartments.py`: Fine-tune the pre-trained GM12878 encoder to predict subcompartments (A1–B3) for a **new cell type** using your own subcompartment labels (see "Fine-tune on a new cell type" below)
- **Pre-trained models**: Model weights in [TECSAS/share/models/](TECSAS/share/models/)
  - `bv_GM12878_155.pt`: GM12878 model trained with 155 experiments at 50kbp resolution (75.8% overall accuracy)
- **Reference data**: Subcompartment annotations and nuclear body association labels (LADs, NADs, SPADs) in [TECSAS/share/](TECSAS/share/)

## Installation

### Requirements

TECSAS requires Python 3.6+ and the following dependencies:

- PyTorch (>=1.7.0)
- NumPy (>=1.18)
- pyBigWig
- requests
- joblib
- tqdm
- urllib3

### Install from PyPI

```bash
pip install TECSAS
```

### Install from source

Clone the repository and install:

```bash
git clone https://github.com/ed29rice/TECSAS.git
cd TECSAS
pip install -e .
```

### Install dependencies

```bash
pip install torch numpy pyBigWig requests joblib tqdm urllib3
```

**Note**: For GPU acceleration, ensure you have CUDA-compatible PyTorch installed



## Quick Start

### Option A: Use pre-trained weights

Pre-trained model weights for GM12878 (155 experiments, 50kbp resolution) are included in [`TECSAS/share/models/`](TECSAS/share/models/). You can load and use them directly without retraining:

```python
import torch
from TECSAS import TECSAS

# Model configuration matching the pre-trained weights
n_neighbors = 14   # Neighboring bins on each side (context window)
n_predict = 3      # Number of loci to predict
NEXP = 155         # Number of experiments in GM12878
nfeatures = NEXP * (2 * n_neighbors + 1)  # 155 * 29 = 4495

model = TECSAS(n_predict, emsize=128, nhead=8, d_hid=64, nlayers=2,
               nfeatures=nfeatures, ostates=5, dropout=0.01)

# Load pre-trained weights (keys have a 'module.' prefix from DataParallel)
state = torch.load('TECSAS/share/models/bv_GM12878_155.pt', map_location='cpu')
model.load_state_dict({'.'.join(k.split('.')[1:]): v for k, v in state.items()})
model.eval()
```

See [`Tutorials/Load_model_GM12878_155exp_50kbp.ipynb`](Tutorials/Load_model_GM12878_155exp_50kbp.ipynb) for a complete evaluation example.

### Option B: Train from scratch

If you want to retrain the model on your own data or a different cell line:

1. **Import TECSAS**:
   ```python
   from TECSAS import data_process, TECSAS
   ```

2. **Download and process epigenomic data from ENCODE**:
   ```python
   dp = data_process(cell_line='GM12878', assembly='hg19', histones=True, tf=True)
   dp.download_and_process_cell_line_data(nproc=10)
   dp.download_and_process_ref_data(nproc=10)
   ```

3. **Generate training data**:
   ```python
   train, val, test, averages, indices = dp.training_data(n_neigbors=14, train_per=0.8)
   ```

4. **Initialize and train the model**:
   ```python
   model = TECSAS(n_predict=3, emsize=128, nhead=8, d_hid=64, nlayers=2,
                  nfeatures=NEXP*(2*14+1), ostates=5, dropout=0.01)
   # ... training loop (see Tutorials/train_and_predict_HistMod_example.ipynb)
   ```

5. **Make predictions on a target cell line**:
   ```python
   test_data = dp.test_set(chr=1)
   predictions = model(test_data, None)[0].argmax(dim=-1)
   ```

See the [Tutorials/](Tutorials/) directory for complete training and prediction workflows.

### Option C: Fine-tune on a new cell type (transfer learning)

To generate subcompartment annotations (A1, A2, B1, B2, B3) for a cell type other than the
bundled ones, you generally **do not need to train from scratch**. Because TECSAS predicts
structure directly from the epigenome, a model trained on a cell line with Hi-C-derived labels
transfers to other cell types. Two cases:

- **No retraining** — if your target cell type has the same panel of experiments the pre-trained
  model expects (155 experiments @ 50 kb for `bv_GM12878_155.pt`), process its tracks and predict
  directly (Option A + `data_process.test_data` / `test_set`).
- **Fine-tune** — if your target has a different experiment set/resolution, or you have your own
  Hi-C-derived subcompartment labels and want cell-type-specific accuracy, fine-tune with transfer
  learning: freeze the pre-trained GM12878 encoder and retrain only the output head. The encoder
  and transformer operate per sequence position, so they transfer even when the number of
  experiments differs; only a fresh head is trained.

A ready-to-run workflow is provided:

```bash
python Tutorials/finetune_subcompartments.py \
    --cell-line MyCell --labels-dir /path/to/my_labels_50kb \
    --pretrained TECSAS/share/models/bv_GM12878_155.pt \
    --output-dir ./finetune_MyCell --epochs 75
```

or use the notebook [`Tutorials/Finetune_subcompartments_new_cell_50kb.ipynb`](Tutorials/Finetune_subcompartments_new_cell_50kb.ipynb).
The target cell line's epigenomic tracks are downloaded automatically from ENCODE; you supply the
**training labels** as a directory of per-chromosome files named `chr{1..22}_beads.txt.original`,
space-delimited with the subcompartment label (`A1/A2/B1/B2/B3`, `NA` allowed) in column 1, one row
per 50 kb bin (matching `data_process.chrm_size`). The shipped `TECSAS/share/subcom_GM12878_50kb/`
files are an example of this exact format. Outputs include the fine-tuned weights, a predictions
`.txt`, and a colored `.bed` track for genome-browser visualization.

## Citation

If you use TECSAS in your research, please cite:

Dodero-Rojas, E., Mendieta, A., Fehlis, Y., Mayala, N., Contessoto, V. G., & Onuchic, J. N. (2025). Epigenetics is all you need: A transformer to decode chromatin structural compartments from the epigenome. *PLOS Computational Biology*, *21*(12), e1012326. [https://doi.org/10.1371/journal.pcbi.1012326](https://doi.org/10.1371/journal.pcbi.1012326)

## License

TECSAS is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This research was supported by the Center for Theoretical Biological Physics, sponsored by the NSF (Grants PHY-2019745 and PHY-2210291) and by the Welch Foundation (Grant C-1792). We thank AMD (Advanced Micro Devices, Inc.) for the donation of critical hardware and support resources from its HPC Fund that made this work possible.

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the developers.

---

Copyright (c) 2020-2025 The Center for Theoretical Biological Physics (CTBP) - Rice University
