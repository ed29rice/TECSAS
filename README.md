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
  - `Load_model_K562_124_exp_25kbp.ipynb`: Load K562 model at 25kb resolution
  - `train_and_predict_XADS_HistMod_RNASeq.ipynb`: Complete workflow for nuclear body association (LADs/NADs/SPADs) prediction using transfer learning
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

1. **Import TECSAS**:
   ```python
   from TECSAS import data_process, TECSAS
   ```

2. **Download data for your cell line**:
   ```python
   dp = data_process(cell_line='GM12878', assembly='hg19', histones=True, tf=True)
   dp.download_and_process_cell_line_data(nproc=10)
   ```

3. **Generate training data**:
   ```python
   train, val, test, averages, indices = dp.training_data(n_neigbors=14, train_per=0.8)
   ```

4. **Train or load a model**:
   ```python
   model = TECSAS(n_predict=3, emsize=128, nhead=8, d_hid=64, nlayers=2,
                  nfeatures=29, ostates=5, dropout=0.01)
   # Train or load pre-trained weights
   model.load_state_dict(torch.load('model.pt'))
   ```

5. **Make predictions**:
   ```python
   test_data, averages = dp.test_set(chr=1)
   predictions = model(test_data, None)[0].argmax(dim=-1)
   ```

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
