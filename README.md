# scDVD

**single-cell Dual-view Diffusion for Perturbation Prediction**

Seunghoo Song and Minsik Oh — Department of Artificial Intelligence, Myongji University

scDVD predicts treated single-cell expression distributions from unpaired control
and treated populations. It models responses relative to a matched training-control
mean and combines a condition-dependent mean, a cell view and a population view.

```text
drug + dose + cell context + batch
                  |
matched controls -> dual-view response diffusion -> restore control anchor
                  |                                         |
       mean + centered cell deviation              clip + empirical mask
            + centered population correction                |
                                                generated treated expression
```

The paper configuration uses **set size 4**, no population-correction RMS cap,
and no population-correction gate. The molecular dose gate and residual-block
modulation gates remain. The objective is positive-expression MSE plus
`0.001 × GeomLoss energy`, with padding excluded from the direct loss weights.
See [the method and implementation map](docs/method.md).

## Install

Python 3.10 is the verified reference interpreter.

```bash
git clone https://github.com/seunghoo123/scDVD.git
cd scDVD
python -m pip install -r requirements/paper.txt
python -m pip install -e '.[data]'
scdvd --help
```

`requirements/paper.txt` records the main package versions from the final-run
environment. Install a PyTorch build suitable for your CPU/CUDA system. For
development outside the reference environment, `python -m pip install -e .`
installs the declared dependencies. The package includes its Hydra configs.

## Run a small example

```bash
python examples/cpu_demo.py --output /tmp/scdvd-demo
```

This synthetic CPU example trains a small diffusion model and mask model, loads
their checkpoints, and evaluates one held-out condition. It writes metrics and
`GT`/`scDVD` expression arrays. It checks the execution pipeline; it is not a
paper experiment. Use a new output directory for each invocation.

## Use the paper datasets

The code bundle contains exact cell assignments, gene orders, feature hashes,
run provenance and recorded scDVD metrics. Large H5AD files, molecular features
and model checkpoints are separate assets. Public download URLs have not yet
been assigned; their current availability is explicit in
[the asset manifest](paper/manifests/assets.json). See [data preparation](docs/data.md)
for the input contract and how to apply the released assignments.

```text
data/
  sciplex3/                   # the same layout also applies to op3/
    features/unimol.npy
    features/drug_lookup.csv
    r1/s1/data.h5ad
    ...                      # three repeats × five repeated holdouts
```

Use `SCDVD_DATA_ROOT` or `paths.data_root=...` to choose the data directory.
All relative paths resolve from the working directory; commands do not change it.

```bash
# Inspect the full configuration without starting an experiment.
scdvd train --config-name sciplex3 --cfg job

# Train diffusion and the independent mask predictor.
scdvd train --config-name sciplex3 paths.data_root=/path/to/data repeat=1 split=1
scdvd train-mask --config-name sciplex3 paths.data_root=/path/to/data repeat=1 split=1

# Evaluate a best-validation checkpoint with its matching mask checkpoint.
scdvd evaluate --config-name sciplex3 paths.data_root=/path/to/data \
  checkpoint=/path/to/diffusion.ckpt mask_checkpoint=/path/to/mask.ckpt \
  evaluation.split=ood evaluation.save_predictions=true
```

Replace `sciplex3` with `op3` for OP3. Use `training.accelerator=cpu`,
`mask_training.accelerator=cpu` or `evaluation.device=cpu` for CPU execution.
The full paper model is substantially larger than the synthetic example.
`python train.py`, `python train_mask.py`, and `python evaluate_diffusion.py`
remain wrappers around the same installed implementations.

Evaluation writes `condition_metrics.csv`, `summary.json`, `summary.csv`,
`resolved_config.yaml`, and checkpoint/feature hashes in `run_provenance.json`.
Optional prediction archives contain `GT`, `scDVD`, gene names and condition
metadata in the processed H5AD expression scale. This benchmark evaluator uses
held-out GT conditions; it is not a standalone arbitrary-condition query interface.

## Reproduce the paper tables

The final cohorts are sciPlex3 **fixed9** and OP3 **repeat7**, both with three
repeats and five repeated condition holdouts. These are not disjoint CV folds.
The exact run identities are in [runs.json](paper/manifests/runs.json).

```bash
python scripts/aggregate_results.py \
  --input paper/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

This recreates the scDVD entries of Tables 1–4 from recorded per-run means;
it does not rerun inference. The reported SD is the sample SD of the three
repeat averages. See [reproduction details](docs/reproduction.md) for training,
mask reuse, evaluation conventions and the scope of figure reproduction.

## Repository layout

```text
src/scdvd/           model, losses, data, sampling, evaluation and CLI
src/scdvd/configs/   portable paper defaults for sciPlex3 and OP3
examples/           synthetic CPU example
tests/              loss, sampling, split isolation and checkpoint tests
paper/              exact splits, manifests and recorded set4 table inputs
scripts/            data preparation, aggregation, compatibility verification, export
docs/               method, data, reproduction and release notes
```

`scdvd/` and `config/` are compatibility links. The local `archive/` contains
historical scripts/configs; local `outputs/`, `data/` and `provenance/` retain
research artifacts. These directories and the supplied manuscript are excluded
from the source-only export.

```bash
python -m unittest discover -s tests -v
python scripts/export_release.py --output ../scDVD_source.tar.gz
```

## Citation and release metadata

Author and title metadata are provided in [CITATION.cff](CITATION.cff).
The source repository is [github.com/seunghoo123/scDVD](https://github.com/seunghoo123/scDVD).
A project license, public data/checkpoint download locations, and publication or
release DOIs have not yet been assigned; see [release notes](docs/release.md).
