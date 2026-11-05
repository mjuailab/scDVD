<h1>
  <img src="docs/assets/scDVD_icon.png" alt="scDVD icon" width="46" align="absmiddle">
  scDVD: single-cell Dual-view Diffusion for Perturbation Prediction
</h1>

**Seunghoo Song** and **Minsik Oh**  
Department of Artificial Intelligence, Myongji University

scDVD predicts treated-cell expression distributions from unpaired control and treated scRNA-seq populations using **control-anchored dual-view diffusion**.

[Overview](#overview) · [Installation](#installation) · [Data](#data) · [Training](#training) · [Evaluation](#evaluation) · [Reproducibility](#reproducibility) · [Documentation](#documentation)

## Overview

<p align="center">
  <img src="docs/assets/scDVD.png" alt="Overview of the scDVD framework" width="850">
</p>

scDVD models responses relative to a fixed training-control mean through a **condition-induced shift**, a **population-informed correction**, and a **cell-specific deviation**. After 50-step DDIM sampling, the control reference is added back and a separate mask predictor supports empirical masking to recover expression sparsity.

**Paper configuration:** K = 4 target cells and K = 4 matched controls per set; population-correction RMS cap and population gate removed. Dose and residual-block modulation gates remain. [Model details](docs/method.md) · [한국어](docs/method_ko.md)

## Installation

Reference environment: **Linux · Python 3.10 · PyTorch 2.1.0 · CUDA 12.1**.

```bash
git clone https://github.com/mjuailab/scDVD.git
cd scDVD
conda env create -f environment.yml
conda activate scdvd
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
scdvd --help
```

For CPU-only installation, alternative environments and troubleshooting, see the [installation guide](docs/installation.md).

### Quick start

Run a synthetic end-to-end example on CPU, without experimental data:

```bash
python scripts/cpu_demo.py --output outputs/cpu_demo
```

This trains both models, reloads checkpoints and evaluates a held-out condition. Outputs include metrics and `GT`/`scDVD` arrays. Use a new output directory for each run; these example metrics are not paper results.

## Data

| Dataset | Source | Benchmark |
|:--|:--|:--|
| **sciPlex3** | [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) | A549, K562, MCF7; 9 fixed unseen drugs |
| **OP3** | [GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) | 6 cell types; 7 unseen drugs per repetition |

Exact split assignments, gene lists, feature hashes and recorded metrics are included in [`benchmarks/`](benchmarks/README.md). Public downloads for the **prepared H5ADs, Uni-Mol features and pretrained checkpoints are not yet available**; raw GEO files require additional preprocessing.

Before paper training or evaluation, follow [Data preparation](docs/data.md) to supply the matching prepared assets and apply a released split with `scripts/prepare_data.py`.

## Training

Train the diffusion model and the separate mask predictor for a prepared split:

```bash
scdvd train --config-name sciplex3 repeat=1 split=1
scdvd train-mask --config-name sciplex3 repeat=1 split=1
```

Use `--config-name op3` for OP3. Both commands print the best-validation checkpoint. Training defaults to one GPU; device selection, data locations and output paths are described in [Running experiments](docs/usage.md).

## Evaluation

Use diffusion and mask checkpoints from the **same dataset, repetition and split**, together with its prepared data and molecular features:

```bash
scdvd evaluate --config-name sciplex3 repeat=1 split=1 \
  checkpoint=/path/to/best_diffusion.ckpt \
  mask_checkpoint=/path/to/best_mask.ckpt \
  evaluation.split=ood evaluation.save_predictions=true
```

`ood` evaluates unseen drugs; `test` evaluates held-out conditions of training drugs. Outputs contain per-condition and summary metrics, optional `GT`/`scDVD` arrays, and run provenance (for example, `outputs/sciplex3_r1_s1/ood/`).

The CLI evaluates conditions with observed GT; arbitrary-condition prediction is not exposed. See [evaluation details](docs/usage.md#evaluate-test-and-ood).

## Reproducibility

Each dataset uses **3 repetitions × 5 random condition-level holdouts**, with set size 4. These are repeated holdouts rather than disjoint cross-validation folds. Exact assignments and run provenance are in [`benchmarks/`](benchmarks/README.md).

Recreate the scDVD summary tables from saved per-run metrics:

```bash
python scripts/aggregate_results.py \
  --input benchmarks/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

This aggregates recorded metrics only. See the [reproduction protocol](docs/reproduction.md) for checkpoint provenance, aggregation and figure coverage.

## Repository structure

```text
scDVD/
├── src/scdvd/    # Model, data loaders, training and evaluation
├── configs/      # Dataset, model and training settings
├── scripts/      # Data preparation, demo, aggregation; dev/ utilities
├── benchmarks/   # Exact splits, run provenance and recorded metrics
├── docs/         # Guides and assets/ images
└── tests/        # Model and data-contract checks
```

Run commands from the repository root after installation. Local `data/` and `outputs/` are excluded from Git.

## Documentation

| Guide | Contents |
|:--|:--|
| [Installation](docs/installation.md) | Environment setup and troubleshooting |
| [Data preparation](docs/data.md) | Sources, input formats and checkpoints |
| [Running experiments](docs/usage.md) | Training, evaluation and configuration |
| [Model architecture](docs/method.md) · [한국어](docs/method_ko.md) | Formulation and implementation |
| [Reproduction protocol](docs/reproduction.md) | Splits, metrics and provenance |
| [Development](docs/development.md) · [Validation](docs/validation.md) | Tests, package builds and validation records |

## Citation

See [CITATION.cff](CITATION.cff) for authors and software metadata. A publication DOI and project license have not yet been assigned.
