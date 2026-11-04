# scDVD

**single-cell Dual-view Diffusion for Perturbation Prediction**

Seunghoo Song and Minsik Oh · Department of Artificial Intelligence, Myongji University

[Installation](#installation) · [Data](#data) · [Training](#training) · [Evaluation](#evaluation) · [Reproducibility](#reproducibility) · [Documentation](#documentation)

scDVD predicts single-cell gene expression responses to drug perturbations from
unpaired control and treated cells. It combines a condition-dependent mean with
cell and population views in a control-anchored diffusion model, followed by
empirical masking to recover expression sparsity.

The paper model uses **set size 4**, with the population-correction RMS cap and
population gate removed. Dose gates and residual-block modulation gates remain.
See [Model architecture](docs/method.md) ([한국어](docs/method_ko.md)) for the
overall flow, output decomposition and implementation details.

## Repository structure

```text
scDVD/
├── configs/             # Dataset, model and training settings
│   ├── sciplex3.yaml
│   ├── op3.yaml
│   ├── dataset/
│   ├── model/
│   └── training/
├── src/scdvd/           # Installable model, data loaders, losses and evaluation
├── scripts/             # Training/evaluation entry points, data preparation, tables
├── examples/            # Small synthetic CPU example
├── benchmarks/          # Exact paper splits, provenance and recorded results
├── docs/                # Installation, data, usage and reproduction guides
├── tests/               # Model and data-contract checks
├── tools/               # Checkpoint compatibility and source export utilities
├── environment.yml      # Conda Python environment
└── requirements.txt     # Reference package versions
```

`data/` and `outputs/` are created locally and excluded from Git. Run the commands
below from the repository root after installing the package.

## Installation

The reference environment is **Linux, Python 3.10 and PyTorch 2.1.0**. The final
experiments used the CUDA 12.1 build of PyTorch.

```bash
git clone https://github.com/mjuailab/scDVD.git
cd scDVD
conda env create -f environment.yml
conda activate scdvd

# Update the installer before resolving the older reference wheels.
python -m pip install --upgrade pip 'setuptools<82' wheel

# GPU environment: install the CUDA 12.1 PyTorch wheel.
python -m pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
scdvd --help
```

For a CPU environment, replace the PyTorch command above with:

```bash
python -m pip install torch==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

Keep PyPI enabled for general dependencies; the explicit `+cpu` / `+cu121`
suffix selects the PyTorch build. Using only the PyTorch index can fail to resolve
build dependencies in a fresh environment.

See [Installation](docs/installation.md) for `venv`, CUDA 11.8, dependency versions
and troubleshooting. Available wheel choices are documented by
[PyTorch](https://pytorch.org/get-started/previous-versions/#v210).

### Quick start: synthetic CPU example

```bash
python examples/cpu_demo.py --output outputs/cpu_demo
```

This trains a small diffusion model and mask predictor, reloads both checkpoints,
and evaluates one synthetic held-out condition. It requires no downloaded data
or GPU. Results appear under `outputs/cpu_demo/evaluation/`, including metrics
and expression arrays named `GT` and `scDVD`. Choose a fresh output directory
when rerunning. This example validates the pipeline; its metrics are not paper results.

To exercise the OP3 column mappings and held-out-drug evaluation instead:

```bash
python examples/cpu_demo.py --config-name op3 --evaluation-split ood --output outputs/op3_demo
```

The default device is CPU. With a CUDA PyTorch build, add `--device cuda` to
exercise GPU training and inference; select a GPU with `CUDA_VISIBLE_DEVICES`.

See the [0.2.1 validation record](docs/validation.md) for fresh-install checks,
CPU/GPU execution and comparisons with the original set4 checkpoints.

## Data

| Dataset | Original data | scDVD experiment |
|---|---|---|
| sciPlex3 | [GEO GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944), sample [GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378) | A549, K562 and MCF7; nine fixed OOD drugs; 3 repeats × 5 splits |
| OP3 | [GEO GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945) ([H5AD download](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279945/suppl/GSE279945_sc_counts_processed.h5ad)) | Six original cell types; seven OOD drugs per repeat; 3 repeats × 5 splits |

**Asset availability.** This repository includes exact cell assignments, ordered
gene lists, feature hashes and recorded metrics. The scDVD-preprocessed H5ADs,
Uni-Mol feature matrices and pretrained checkpoints do **not yet have public
download links**. Original GEO files need further preprocessing and cannot be
passed directly to the paper training commands. See [Data preparation](docs/data.md)
for sources, preprocessing conventions and the required input files.

If you already have the matching prepared data and features, create one split:

```bash
python scripts/prepare_data.py --dataset sciplex3 \
  --h5ad /path/to/preprocessed_all_cells.h5ad \
  --features /path/to/unimol.npy \
  --lookup /path/to/drug_lookup.csv \
  --repeat 1 --split 1 --output data
```

This validates the prepared inputs and applies the released cell assignment;
it does not perform raw-data preprocessing. The resulting layout is:

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

After preparing a split, train the diffusion model and the separate mask predictor:

```bash
scdvd train --config-name sciplex3 repeat=1 split=1
scdvd train-mask --config-name sciplex3 repeat=1 split=1
```

Replace `sciplex3` with `op3` for OP3. Both stages print the path of the checkpoint
with the lowest validation loss. Default checkpoint directories are:

```text
outputs/sciplex3_r1_s1_set4/   # Diffusion checkpoints and training metadata
outputs/sciplex3_r1_s1_mask/   # Mask checkpoints and training metadata
```

The defaults use one GPU. To select a device, prefix a command with
`CUDA_VISIBLE_DEVICES=0`. To use another data location, set
`SCDVD_DATA_ROOT=/absolute/path/to/data` or pass `paths.data_root=...`.
See [Running experiments](docs/usage.md) for CPU overrides, custom output paths,
configuration changes and all-repeat execution.

## Evaluation

Use the best-validation diffusion checkpoint and the mask checkpoint from the
**same dataset, repeat and split**. Substitute the paths printed during training:

```bash
scdvd evaluate --config-name sciplex3 repeat=1 split=1 \
  checkpoint=/path/to/best_diffusion.ckpt \
  mask_checkpoint=/path/to/best_mask.ckpt \
  evaluation.split=ood evaluation.save_predictions=true
```

Use `evaluation.split=test` for held-out conditions of training drugs. OOD
evaluation uses held-out drugs. Outputs are written to
`outputs/sciplex3_r1_s1/ood/` (or `test/`):

| Output | Contents |
|---|---|
| `condition_metrics.csv` | Per-condition ALL, DE20 and DE40 metrics |
| `summary.csv`, `summary.json` | Condition-averaged evaluation metrics |
| `predictions/*.npz` | Optional `GT` and `scDVD` arrays with gene/condition metadata |
| `resolved_config.yaml`, `run_provenance.json` | Configuration, input paths and artifact hashes |

Evaluation requires the prepared H5AD, molecular features and matching mask
checkpoint in addition to the diffusion checkpoint. It evaluates conditions with
observed GT; arbitrary-condition prediction is not exposed by this CLI.

## Reproducibility

The final benchmark uses **sciPlex3 fixed9** and **OP3 repeat7**, each with three
repeats and five repeated condition holdouts. These are not disjoint CV folds.
Both use the canonical set4 model. Exact split assignments and run identities
are in [`benchmarks/`](benchmarks/README.md).

Recreate the scDVD table entries from the included per-run metrics:

```bash
python scripts/aggregate_results.py \
  --input benchmarks/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

This aggregates recorded results without training or inference. For exact model
settings, mask checkpoint reuse, metric definitions and the scope of figure
reproduction, see [Reproduction protocol](docs/reproduction.md).

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | Python environment, CPU/CUDA wheels, troubleshooting |
| [Data preparation](docs/data.md) | Original sources, asset availability, H5AD and feature format |
| [Running experiments](docs/usage.md) | Configurations, training, evaluation, output paths |
| [Model architecture](docs/method.md) · [한국어](docs/method_ko.md) | Diagram, mean/cell/population pathways, objective, generation and masking |
| [Reproduction protocol](docs/reproduction.md) | Final experiments, aggregation and checkpoint compatibility |
| [Development and releases](docs/development.md) | Tests, package builds and source export |
| [Validation record](docs/validation.md) | Tested environments, execution checks and checkpoint comparisons |

## Citation

Author, title and software metadata are provided in [CITATION.cff](CITATION.cff).
A publication DOI and project license have not yet been assigned.
