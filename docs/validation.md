# Installation and execution validation

[Back to README](../README.md)

Version **0.2.1** was checked on **2026-11-05**, starting from a fresh clone of
`mjuailab/scDVD` at `ac7b6b00101974b38a589708242a48fa0a8ffe28`.

## Environment and installation fixes

Two independent Python 3.10.14 `venv` environments were installed without system
site packages: PyTorch **2.1.0+cpu** and **2.1.0+cu121**. GPU checks used an NVIDIA
RTX 4090. Both environments used the reference `requirements.txt`, including
Lightning 2.2.4, NumPy 1.26.4 and setuptools 69.5.1; `pip check` passed.

Fresh installation exposed two failures that an existing shared environment had
hidden: the PyTorch-only package index could not resolve a build dependency,
and newer setuptools no longer provided the `pkg_resources` import required by
the reference Lightning utilities. Installation commands now update pip, select
an explicit CPU/CUDA wheel while keeping PyPI available, and constrain setuptools.
Configuration subpackages are explicitly packaged, and the legacy `config`
symlink now resolves to `configs/`.

These checks used the documented `venv` route. The Conda environment-creation
step, CUDA 11.8 alternative and other operating systems were not exercised.

## Completed checks

| Check | Result |
|---|---|
| Unit tests | 14 passed in both editable and wheel installations |
| Installed CLI/configurations | All six dataset/stage combinations resolved outside the checkout |
| Synthetic pipeline | Diffusion training, mask training, checkpoint reload and evaluation passed on CPU and CUDA for sciPlex3 TEST and OP3 OOD mappings |
| Wheel and source distribution | Built successfully; all six YAML files and model modules included |
| Prepared data | `prepare_data.py` completed for both datasets, repeat 1 / split 1, using the matching local paper assets |
| Full-size GPU training | Both 2,000-gene models completed two diffusion training batches and a short mask-training epoch, with finite validation scores and checkpoints |
| Existing set4 checkpoints | Diffusion and matching mask SHA-256 hashes verified against `benchmarks/manifests/runs.json` |
| Frozen-source compatibility | Parameter keys, initialization and seeded CPU DDIM outputs matched exactly for both datasets; maximum output difference 0 |
| Recorded table aggregation | All 36 entries reproduced; maximum floating-point difference below 1e-18 |

Full-size training checks capped training samples at 128 and used one epoch.
They verify execution and checkpoint writing, not convergence or benchmark
performance. The numerical model, loss, metric implementations and benchmark
configurations are unchanged in this maintenance update.

## Actual-data inference

The verified canonical set4 r1/s1 diffusion checkpoints and their matching masks
were evaluated with **50 DDIM steps**, the prepared data and all observed cells
of the first eligible TEST/OOD condition. Population RMS caps and population
gates are disabled; dose and residual-block gates remain.

| Dataset / split | Cell type | Drug | Model dose | GT / generated cells |
|---|---|---|---:|---:|
| sciPlex3 TEST | A549 | 2-Methoxyestradiol | 10 | 166 / 166 |
| sciPlex3 OOD | A549 | Alvespimycin | 0.01 | 230 / 230 |
| OP3 TEST | B cells | AMD-070 (hydrochloride) | 1 | 219 / 219 |
| OP3 OOD | B cells | Ganetespib (STA-9090) | 1 | 239 / 239 |

All saved `GT`/`scDVD` arrays had matching cell counts and 2,000 genes. Generated
values and metrics were finite. For each of these four conditions, RMSE, energy
distance and EMD on ALL/DE20/DE40 genes matched the original completed run's
recorded condition metrics exactly.

This is a limited execution and compatibility check, not a rerun of all 30
benchmark experiments. Full prepared datasets, Uni-Mol features and pretrained
checkpoints were available locally for this audit; their public download links
remain unassigned, as described in [Data preparation](data.md). The synthetic
examples and recorded-table aggregation work from the public repository alone
after installation.
