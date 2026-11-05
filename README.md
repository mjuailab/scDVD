<h1>
  <img src="assets/scDVD_icon.png" alt="scDVD icon" width="46" align="absmiddle">
  scDVD: single-cell Dual-view Diffusion for Perturbation Prediction
</h1>

**Seunghoo Song** and **Minsik Oh**  
Department of Artificial Intelligence, Myongji University

scDVD is a **control-anchored dual-view diffusion framework** for predicting treated-cell expression distributions from unpaired control and treated scRNA-seq populations. It models perturbation responses relative to a fixed control reference, accounting for condition-induced changes and cell-to-cell variation.

[Overview](#overview) · [Installation](#installation) · [Data](#data) · [Training](#training) · [Evaluation](#evaluation) · [Reproducibility](#reproducibility) · [Documentation](#documentation)

## Overview

<p align="center">
  <img src="assets/scDVD.png" alt="Overview of the scDVD framework" width="850">
</p>

Rather than generating absolute expression directly, scDVD decomposes each **predicted control-relative response** into three complementary components:

| Component | Role |
|:--|:--|
| **Condition-induced shift** | Predicts the gene-wise change shared by all target cells under a given perturbation condition. |
| **Population-informed correction** | Uses joint attention among noisy target cells and sampled controls from the same cellular context and batch. |
| **Cell-specific deviation** | Processes each noisy target cell independently to capture individual differences without cross-cell attention. |

A **fixed control reference** is defined as the gene-wise mean of the corresponding training controls for each cellular-context–batch pair. The two cell-level components are **set-centered** so that they do not change the condition-induced shift in the predicted response mean *before output processing*. The reference is added back after diffusion to recover treated-cell expression.

Training combines positive-expression reconstruction with a set-level energy discrepancy. At inference, **50-step DDIM sampling** generates response sets from Gaussian noise. A **separate mask predictor** estimates gene-presence probabilities, and empirical masks sampled from a matching training condition are applied to the generated expression profiles.

**Paper configuration:** Each set contains **K = 4 target cells and K = 4 matched controls**. The population-correction RMS cap and population gate are removed; dose gates and residual-block modulation gates remain. See [Model architecture](docs/method.md) ([한국어](docs/method_ko.md)) for the complete formulation and implementation details.

## Installation

**Reference environment:** Linux · Python 3.10 · PyTorch 2.1.0 · CUDA 12.1 (GPU experiments).

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/mjuailab/scDVD.git
cd scDVD
conda env create -f environment.yml
conda activate scdvd
```

Install PyTorch and the remaining dependencies:

```bash
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
scdvd --help
```

For a **CPU-only** environment, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

The `+cu121` and `+cpu` suffixes select the PyTorch build; keep PyPI enabled to resolve other dependencies. See the [installation guide](docs/installation.md) for alternative environments, CUDA 11.8, and troubleshooting, or the [PyTorch archive](https://pytorch.org/get-started/previous-versions/#v210) for available wheel variants.

### Quick start: synthetic CPU example

Run a small end-to-end example without experimental data or a GPU:

```bash
python examples/cpu_demo.py --output outputs/cpu_demo
```

The example trains a diffusion model and a mask predictor, reloads their checkpoints, and evaluates a synthetic held-out condition. Metrics and expression arrays (`GT`, `scDVD`) are saved under `outputs/cpu_demo/evaluation/`.

For a synthetic OP3-style example with unseen-drug evaluation:

```bash
python examples/cpu_demo.py --config-name op3 --evaluation-split ood --output outputs/op3_demo
```

The default device is CPU. To use a GPU, add `--device cuda` with a CUDA-enabled PyTorch installation; `CUDA_VISIBLE_DEVICES` selects the device. Use a fresh output directory when rerunning.

> **Note:** These examples validate the execution pipeline; their metrics are not the paper results. See the [validation record](docs/validation.md) for environment and checkpoint checks.

## Data

The experiments use two public single-cell drug perturbation datasets:

| Dataset | Source | Experimental setting |
|:--|:--|:--|
| **sciPlex3** | [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) · [GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378) | A549, K562, and MCF7; 9 fixed unseen drugs; 3 repetitions × 5 splits |
| **OP3** | [GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) · [Processed H5AD](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279945/suppl/GSE279945_sc_counts_processed.h5ad) | 6 cell types; 7 unseen drugs per repetition; 3 repetitions × 5 splits |

**Data availability:** The repository provides exact cell assignments, ordered gene lists, feature hashes, and recorded metrics. Public download links are **not yet available** for the scDVD-preprocessed H5AD files, Uni-Mol feature matrices, or pretrained checkpoints. The original GEO data require additional preprocessing and cannot be passed directly to the paper training commands. See [Data preparation](docs/data.md) for the required inputs and conventions.

### Prepare an experiment split

Given matching preprocessed expression data and drug features, create one split:

```bash
python scripts/prepare_data.py --dataset sciplex3 \
  --h5ad /path/to/preprocessed_all_cells.h5ad \
  --features /path/to/unimol.npy \
  --lookup /path/to/drug_lookup.csv \
  --repeat 1 --split 1 --output data
```

This script validates the input files and applies the released cell assignments; **it does not preprocess raw GEO data**. The resulting layout is:

```text
data/
├── sciplex3/
│   ├── features/
│   │   ├── unimol.npy
│   │   └── drug_lookup.csv
│   └── r1/s1/data.h5ad
└── op3/                 # Same layout
```

## Training

Train the **response diffusion model** and the **separate mask predictor** for a prepared split:

```bash
scdvd train --config-name sciplex3 repeat=1 split=1
scdvd train-mask --config-name sciplex3 repeat=1 split=1
```

Use `--config-name op3` for OP3. Each command prints the checkpoint with the lowest validation loss. By default, checkpoints and metadata are saved to:

```text
outputs/sciplex3_r1_s1_set4/  # Diffusion model
outputs/sciplex3_r1_s1_mask/  # Mask predictor
```

Training uses one GPU by default. Select a GPU with `CUDA_VISIBLE_DEVICES=0`; change the data root with `SCDVD_DATA_ROOT=/absolute/path/to/data` or `paths.data_root=...`. See [Running experiments](docs/usage.md) for device overrides, output paths, and all-repeat execution.

## Evaluation

Use the **best-validation diffusion checkpoint** and a **mask checkpoint from the same dataset, repetition, and split**:

```bash
scdvd evaluate --config-name sciplex3 repeat=1 split=1 \
  checkpoint=/path/to/best_diffusion.ckpt \
  mask_checkpoint=/path/to/best_mask.ckpt \
  evaluation.split=ood evaluation.save_predictions=true
```

Use `evaluation.split=ood` for **unseen-drug** evaluation and `evaluation.split=test` for **held-out-condition** evaluation on training drugs. Outputs are saved under `outputs/sciplex3_r1_s1/ood/` or `outputs/sciplex3_r1_s1/test/`.

| Output | Contents |
|:--|:--|
| `condition_metrics.csv` | Per-condition metrics for ALL, DE20, and DE40 |
| `summary.csv`, `summary.json` | Condition-averaged evaluation metrics |
| `predictions/*.npz` | Optional `GT` and `scDVD` expression arrays with gene/condition metadata |
| `resolved_config.yaml`, `run_provenance.json` | Resolved settings, input paths, and artifact hashes |

Evaluation requires prepared H5AD data, molecular features, and both matching checkpoints. The current CLI evaluates conditions with observed ground truth; it does **not** expose arbitrary-condition prediction.

## Reproducibility

The reported benchmark uses **3 repetitions × 5 random condition-level holdouts** per dataset (15 runs), not disjoint cross-validation folds. The unseen-drug setup uses **9 fixed drugs for sciPlex3** and **7 drugs selected per repetition for OP3**. All main experiments use **K = 4**.

Exact split assignments and run identities are available under [`benchmarks/`](benchmarks/README.md). To recreate the scDVD summary tables from recorded per-run metrics:

```bash
python scripts/aggregate_results.py \
  --input benchmarks/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

This command aggregates **saved metrics only**; it does not train models or generate predictions. See the [reproduction protocol](docs/reproduction.md) for checkpoint usage, evaluation definitions, and the scope of figure reproduction.

## Repository structure

```text
scDVD/
├── assets/             # README icon and overview figure
├── configs/            # Dataset, model, and training configurations
│   ├── sciplex3.yaml
│   ├── op3.yaml
│   ├── dataset/
│   ├── model/
│   └── training/
├── src/scdvd/          # Model, data loaders, losses, and evaluation
├── scripts/            # Data preparation, training, and evaluation tools
├── examples/           # Synthetic CPU example
├── benchmarks/         # Paper split assignments and recorded results
├── docs/               # Usage and reproduction guides
├── tests/              # Model and data-contract checks
├── tools/              # Checkpoint compatibility and source export
├── environment.yml     # Conda environment
└── requirements.txt    # Reference dependency versions
```

`data/` and `outputs/` are created locally and excluded from Git. Unless otherwise specified, run the commands above from the repository root.

## Documentation

| Guide | Contents |
|:--|:--|
| [Installation](docs/installation.md) | Environment setup, CPU/CUDA builds, troubleshooting |
| [Data preparation](docs/data.md) | Data sources, preprocessing conventions, input formats |
| [Running experiments](docs/usage.md) | Configurations, training, evaluation, output paths |
| [Model architecture](docs/method.md) · [한국어](docs/method_ko.md) | Fixed control reference, three-component decomposition, diffusion, sparsity modeling |
| [Reproduction protocol](docs/reproduction.md) | Splits, metrics, aggregation, checkpoint compatibility |
| [Development and releases](docs/development.md) | Tests, package builds, source export |
| [Validation record](docs/validation.md) | Environment validation and checkpoint checks |

## Citation

Author, title, and software metadata are provided in [CITATION.cff](CITATION.cff). A publication DOI and project license have not yet been assigned.
