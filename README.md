<h1>
  <img src="assets/scDVD_icon.png" alt="scDVD icon" width="48" align="absmiddle">
  scDVD : single-cell Dual-view Diffusion for Perturbation Prediction
</h1>

**Seunghoo Song** and **Minsik Oh**  
Department of Artificial Intelligence, Myongji University

scDVD predicts single-cell gene expression responses to drug perturbations using **unpaired control and treated scRNA-seq populations**. Its control-anchored diffusion framework combines condition-level changes with cell-specific and population-informed variation to generate heterogeneous treated-cell expression distributions.

**Navigate:** [Overview](#overview) · [Installation](#installation) · [Data](#data) · [Training](#training) · [Evaluation](#evaluation) · [Reproducibility](#reproducibility) · [Documentation](#documentation)


## Overview

scDVD models the **control-relative perturbation response** through three complementary components:

<p align="center">
  <img src="assets/scDVD.png" alt="scDVD model architecture" width="850">
</p>


| Component | Role |
|---|---|
| **Condition-induced shift** | Captures the transcriptional change shared across cells under a given condition. |
| **Population-informed correction** | Uses interactions among target and matched control cells to refine predictions. |
| **Cell-specific deviation** | Models individual expression differences without cross-cell attention. |

The predicted response is added to a matched control reference to obtain treated-cell expression. A separate empirical mask predictor restores expression sparsity.

**Paper configuration:** Set size **K = 4**. The population-correction RMS cap and population gate are removed; dose gates and residual-block modulation gates remain. For the full architecture and implementation details, see [Model architecture](docs/method.md) ([한국어](docs/method_ko.md)).

## Installation

**Reference environment:** Linux · Python 3.10 · PyTorch 2.1.0 · CUDA 12.1 (GPU experiments).

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/mjuailab/scDVD.git
cd scDVD
conda env create -f environment.yml
conda activate scdvd
```

Install the reference PyTorch build and the remaining dependencies:

```bash
python -m pip install --upgrade pip 'setuptools<82' wheel
python -m pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
scdvd --help
```

**CPU only:** Replace the PyTorch installation command above with:

```bash
python -m pip install torch==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

The explicit `+cu121` / `+cpu` suffix selects the PyTorch build. Keep PyPI enabled for other dependencies; using only the PyTorch index can prevent build dependencies from resolving.

See the [installation guide](docs/installation.md) for `venv`, CUDA 11.8, dependency versions, and troubleshooting. Available wheel variants are listed in the [PyTorch archive](https://pytorch.org/get-started/previous-versions/#v210).

### Quick start: synthetic CPU example

Run the end-to-end pipeline **without downloading experimental data or using a GPU**:

```bash
python examples/cpu_demo.py --output outputs/cpu_demo
```

This example trains a small diffusion model and mask predictor, reloads the checkpoints, and evaluates a synthetic held-out condition. Results are saved in `outputs/cpu_demo/evaluation/`, including metrics and expression arrays named `GT` and `scDVD`.

To test the OP3 column mappings and held-out-drug evaluation:

```bash
python examples/cpu_demo.py --config-name op3 --evaluation-split ood --output outputs/op3_demo
```

The demo defaults to CPU. With a CUDA-enabled PyTorch build, add `--device cuda` to run it on a GPU; use `CUDA_VISIBLE_DEVICES` to select a device. Choose a new output directory for each rerun.

> The demo validates the execution pipeline; its metrics are **not** the paper results. Fresh-install and CPU/GPU checks are documented in the [validation record](docs/validation.md).

## Data

The experiments use two publicly available single-cell drug perturbation datasets:

| Dataset | Original data | Experiment setting |
|---|---|---|
| **sciPlex3** | [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) · [Sample GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378) | A549, K562, MCF7; 9 fixed OOD drugs; 3 repeats × 5 splits |
| **OP3** | [GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) · [Processed H5AD](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279945/suppl/GSE279945_sc_counts_processed.h5ad) | 6 original cell types; 7 OOD drugs per repeat; 3 repeats × 5 splits |

**Data availability:** The repository contains exact cell assignments, ordered gene lists, feature hashes, and recorded metrics. However, public download links are **not yet available** for the scDVD-preprocessed H5AD files, Uni-Mol feature matrices, or pretrained checkpoints. Original GEO files require additional preprocessing and cannot be used directly with the paper training commands. See [Data preparation](docs/data.md) for the expected inputs and conventions.

### Prepare an experiment split

If the matching preprocessed expression data and molecular features are available, create a split using:

```bash
python scripts/prepare_data.py --dataset sciplex3 \
  --h5ad /path/to/preprocessed_all_cells.h5ad \
  --features /path/to/unimol.npy \
  --lookup /path/to/drug_lookup.csv \
  --repeat 1 --split 1 --output data
```

The script validates the inputs and applies the released cell assignments; **it does not preprocess raw GEO data**. The resulting files follow this structure:

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

After preparing a split, train the diffusion model and separate mask predictor:

```bash
scdvd train --config-name sciplex3 repeat=1 split=1
scdvd train-mask --config-name sciplex3 repeat=1 split=1
```

Use `--config-name op3` for OP3. Both commands report the checkpoint with the lowest validation loss. The default output directories are:

```text
outputs/sciplex3_r1_s1_set4/   # Diffusion checkpoints and metadata
outputs/sciplex3_r1_s1_mask/   # Mask checkpoints and metadata
```

Training uses one GPU by default. To select a GPU, prefix the command with `CUDA_VISIBLE_DEVICES=0`. To change the data location, set `SCDVD_DATA_ROOT=/absolute/path/to/data` or pass `paths.data_root=...`.

For alternative devices, output paths, and running all repeats, see [Running experiments](docs/usage.md).

## Evaluation

Evaluate with the **best-validation diffusion checkpoint** and the **mask checkpoint from the same dataset, repeat, and split**:

```bash
scdvd evaluate --config-name sciplex3 repeat=1 split=1 \
  checkpoint=/path/to/best_diffusion.ckpt \
  mask_checkpoint=/path/to/best_mask.ckpt \
  evaluation.split=ood evaluation.save_predictions=true
```

Use `evaluation.split=ood` for held-out drugs or `evaluation.split=test` for held-out conditions of training drugs.

Results are saved to `outputs/sciplex3_r1_s1/ood/` (or `test/`):

| Output | Description |
|---|---|
| `condition_metrics.csv` | Per-condition metrics for ALL, DE20, and DE40 |
| `summary.csv`, `summary.json` | Condition-averaged metrics |
| `predictions/*.npz` | Optional `GT` and `scDVD` arrays with gene/condition metadata |
| `resolved_config.yaml`, `run_provenance.json` | Configuration, input paths, and artifact hashes |

Evaluation requires prepared H5AD data, molecular features, and the matching mask checkpoint in addition to the diffusion checkpoint. The current CLI evaluates conditions with observed ground truth; arbitrary-condition prediction is not exposed through this command.

## Reproducibility

The final benchmark uses **sciPlex3 fixed9** and **OP3 repeat7**, with **3 repetitions × 5 random condition-level holdouts** for each dataset. These are repeated holdouts, **not disjoint cross-validation folds**. The canonical model uses **K = 4**.

Exact split assignments and run identities are provided in [`benchmarks/`](benchmarks/README.md). To aggregate the included per-run metrics into the scDVD table entries:

```bash
python scripts/aggregate_results.py \
  --input benchmarks/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

This command aggregates saved evaluation results; it does **not** run training or inference. See the [reproduction protocol](docs/reproduction.md) for model settings, mask checkpoint reuse, metric definitions, and the scope of figure reproduction.

## Repository structure

```text
scDVD/
├── configs/            # Dataset, model, and training configurations
│   ├── sciplex3.yaml
│   ├── op3.yaml
│   ├── dataset/
│   ├── model/
│   └── training/
├── src/scdvd/          # Model, data loaders, losses, and evaluation
├── scripts/            # Training, evaluation, and data preparation
├── examples/           # Synthetic CPU example
├── benchmarks/         # Paper splits and recorded results
├── docs/               # Guides and reproduction details
├── tests/              # Model and data-contract checks
├── tools/              # Checkpoint compatibility and source export
├── environment.yml     # Conda environment
└── requirements.txt    # Reference dependency versions
```

The `data/` and `outputs/` directories are created locally and excluded from Git. Unless otherwise noted, run commands from the repository root.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | Environment setup, CPU/CUDA builds, troubleshooting |
| [Data preparation](docs/data.md) | Sources, preprocessing conventions, required inputs |
| [Running experiments](docs/usage.md) | Configurations, training, evaluation, outputs |
| [Model architecture](docs/method.md) · [한국어](docs/method_ko.md) | Architecture, response decomposition, objective, masking |
| [Reproduction protocol](docs/reproduction.md) | Evaluation splits, aggregation, checkpoint compatibility |
| [Development and releases](docs/development.md) | Tests, package builds, source export |
| [Validation record](docs/validation.md) | Tested environments, execution and checkpoint checks |

## Citation

Author, title, and software metadata are available in [CITATION.cff](CITATION.cff). A publication DOI and project license have not yet been assigned.
