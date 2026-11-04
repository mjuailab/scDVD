# Installation

[Back to README](../README.md)

## Reference environment

| Component | Reference version |
|---|---|
| Operating system | Linux |
| Python | 3.10 (final runs: 3.10.14) |
| PyTorch | 2.1.0, CUDA 12.1 build for the final runs |
| Lightning | 2.2.4 |
| NumPy / SciPy | 1.26.4 / 1.13.0 |
| AnnData / h5py | 0.10.7 / 3.11.0 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |
| GeomLoss | 0.2.6 |

[`requirements.txt`](../requirements.txt) records the principal package versions
from the final-run environment. It is a reference requirements list, not a lock
of every transitive dependency. CPU checks exercise the same pipeline; full
paper training uses a GPU. Other operating systems and newer dependency
combinations have not been validated for paper reproduction.

## 1. Create an isolated environment

Clone the repository and enter it:

```bash
git clone https://github.com/mjuailab/scDVD.git
cd scDVD
```

With Conda:

```bash
conda env create -f environment.yml
conda activate scdvd
```

Or, with an existing Python 3.10 interpreter:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

`environment.yml` creates the Python environment only. Continue with PyTorch
and the Python dependencies below.

## 2. Install one PyTorch build

Choose **one** command for your machine:

```bash
# CUDA 12.1 (final-run reference)
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

```bash
# CUDA 11.8 (alternative wheel; not the final-run reference)
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

```bash
# CPU only (example and tests)
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

These wheel locations are documented by
[PyTorch](https://pytorch.org/get-started/previous-versions/#v210). For GPU use,
select a build supported by your NVIDIA driver. scDVD does not require
`torchvision` or `torchaudio`.

## 3. Install scDVD

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
scdvd --version
scdvd --help
```

The first command installs the reference dependencies, including data-preparation
libraries. The second installs scDVD in editable mode so local source and config
changes take effect. For a regular installation use `python -m pip install --no-deps .`.
For development with unpinned dependencies, `python -m pip install -e '.[data,dev]'`
uses the ranges in `pyproject.toml`; this is not the reference environment.

## 4. Check the installation

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
scdvd train --config-name sciplex3 --cfg job
python examples/cpu_demo.py --output outputs/cpu_demo
```

`--cfg job` prints the configuration without loading data or starting training.
The example explicitly uses the CPU, including when a GPU is available. It
requires a new output directory and reports `CPU demo passed` on success.

## Troubleshooting

| Symptom | Action |
|---|---|
| `scdvd: command not found` | Activate the installed environment; try `python -m scdvd --help`. |
| Missing `scdvd.configs` | Reinstall with `python -m pip install --no-deps -e .`. Configs are installed from the root `configs/` directory; setting `PYTHONPATH=src` alone is insufficient. |
| Missing `geomloss` or `anndata` | Install `requirements.txt` in the same environment used to run scDVD. |
| NumPy ABI/import error with old PyTorch | Use the pinned `numpy==1.26.4` and a clean Python 3.10 environment. |
| CUDA unavailable | Check the selected PyTorch wheel and driver; use the CPU demo to verify the pipeline independently. |
| Data file not found | Prepare the dataset and set `SCDVD_DATA_ROOT`; paths are relative to your working directory by default. |
| Example output already exists | Choose a new `--output` path. The example does not overwrite an earlier run. |
