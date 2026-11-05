# Development and releases

[Back to README](../README.md)

## Source layout

Model, loss, dataset and evaluation code live in `src/scdvd/`. YAML configuration
files live in the top-level `configs/` directory and are installed as the
`scdvd.configs` package. Hydra loads them through that package, so both editable
installs and built wheels use the same configurations. After adding a new Python
subpackage, update the explicit package list in `pyproject.toml`.

Training and evaluation use the installed `scdvd` command. Data preparation,
table aggregation and the synthetic example live in `scripts/`; release and
checkpoint utilities live in `scripts/dev/`. Documentation and README images
live in `docs/`, with images under `docs/assets/`. `benchmarks/` contains exact
splits and recorded metrics. Keep large datasets, checkpoints and experiment
outputs outside the tracked source tree.

## Checks

After following [Installation](installation.md):

```bash
python -m unittest discover -s tests -v
scdvd train --config-name sciplex3 --cfg job
scdvd evaluate --config-name op3 --cfg job
python scripts/cpu_demo.py --output outputs/cpu_demo_check
python scripts/cpu_demo.py --config-name op3 --evaluation-split ood --output outputs/op3_demo_check
```

The tests cover loss values, balanced sampling, split isolation, padding,
checkpoint loading, data assignments and table aggregation. The synthetic CPU
example exercises diffusion training, mask training and masked evaluation, checking
optimizer steps, finite validation scores, checkpoint tensors and prediction metrics.
Both dataset mappings are exercised in CI. Add `--device cuda` to explicitly run
the example on a GPU; CPU remains the default. GitHub Actions also checks dependency
consistency and loads installed configurations outside the checkout with CPU PyTorch.

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
python scripts/dev/export_release.py --output ../scDVD_source.tar.gz
```

The exporter includes source, configurations, requirements, documentation and
images, scripts, tests and benchmark definitions. It excludes local data, checkpoints,
outputs, private manifests and the manuscript PDF. It refuses to overwrite an
existing archive and does not upload it.

## Layout migration

| Previous path | Current path |
|---|---|
| `assets/` | `docs/assets/` |
| `examples/cpu_demo.py` | `scripts/cpu_demo.py` |
| `tools/` | `scripts/dev/` |
| `scripts/train.py`, `scripts/train_mask.py`, `scripts/evaluate.py` | `scdvd train`, `scdvd train-mask`, `scdvd evaluate` |
| Root `scdvd` and `config` compatibility links | Install the package; source lives in `src/scdvd/` and settings in `configs/` |

The `scdvd train`, `scdvd train-mask` and `scdvd evaluate` commands are unchanged.
Model module paths, checkpoint parameter names, numerical settings and recorded
benchmark files are retained. Use an installed package (editable or wheel) for
imports; the repository root no longer provides import/configuration aliases.

## Release metadata

[`CITATION.cff`](../CITATION.cff) contains the known authors, title, software version
and repository URL. A project license, publication DOI and public download links
for the prepared data/features/checkpoints have not yet been assigned. The
original dataset and third-party software references are listed in the data and
installation guides.
