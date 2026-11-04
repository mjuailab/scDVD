# Development and releases

[Back to README](../README.md)

## Source layout

Model, loss, dataset and evaluation code live in `src/scdvd/`. YAML configuration
files live in the top-level `configs/` directory and are installed as the
`scdvd.configs` package. Hydra loads them through that package, so both editable
installs and built wheels use the same configurations. After adding a new Python
subpackage, update the explicit package list in `pyproject.toml`.

User entry points live in `scripts/`; maintenance utilities live in `tools/`.
`benchmarks/` contains exact splits and recorded metrics. Keep large datasets,
checkpoints and experiment outputs outside the tracked source tree.

## Checks

After following [Installation](installation.md):

```bash
python -m unittest discover -s tests -v
scdvd train --config-name sciplex3 --cfg job
scdvd evaluate --config-name op3 --cfg job
python examples/cpu_demo.py --output outputs/cpu_demo_check
```

The tests cover loss values, balanced sampling, split isolation, padding,
checkpoint loading, data assignments and table aggregation. The synthetic CPU
example exercises diffusion training, mask training and masked evaluation.
GitHub Actions runs these checks against an installed package with CPU PyTorch.

## Build a Python package

```bash
python -m pip install build
python -m build
```

The wheel includes model code and YAML configurations. The repository's benchmark
assignments, data-preparation scripts and documentation remain in the source
checkout; a wheel alone is not the full benchmark bundle.

## Export the complete source bundle

```bash
python tools/export_release.py --output ../scDVD_source.tar.gz
```

The exporter includes source, configurations, requirements, documentation,
examples, tests and benchmark definitions. It excludes local data, checkpoints,
outputs, private manifests and the manuscript PDF. It refuses to overwrite an
existing archive and does not upload it.

## Layout migration

| Previous path | Current path |
|---|---|
| `src/scdvd/configs/` | `configs/` (still installed as `scdvd.configs`) |
| `paper/` | `benchmarks/` |
| `requirements/paper.txt` | `requirements.txt` |
| `train.py`, `train_mask.py`, `evaluate_diffusion.py` | `scripts/train.py`, `scripts/train_mask.py`, `scripts/evaluate.py` |
| `scripts/verify_frozen_compatibility.py` | `tools/verify_frozen_compatibility.py` |
| `scripts/export_release.py` | `tools/export_release.py` |

The `scdvd train`, `scdvd train-mask` and `scdvd evaluate` commands are unchanged.
Model module paths, checkpoint parameter names, numerical settings and recorded
benchmark files are retained. Reinstall editable packages after updating the
checkout so the relocated configuration package is registered.

## Release metadata

[`CITATION.cff`](../CITATION.cff) contains the known authors, title, software version
and repository URL. A project license, publication DOI and public download links
for the prepared data/features/checkpoints have not yet been assigned. The
original dataset and third-party software references are listed in the data and
installation guides.
