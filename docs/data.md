# Data preparation and checkpoints

[Back to README](../README.md) · [Running experiments](usage.md)

## Original data sources

| Dataset | Source used for scDVD | Additional reference |
|---|---|---|
| sciPlex3 | [GEO GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944), sample [GSM4150378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378) | [Original sci-Plex code](https://github.com/cole-trapnell-lab/sci-plex) |
| OP3 | [GEO GSE279945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279945), `GSE279945_sc_counts_processed.h5ad` | [Open Problems competition description](https://openproblems.bio/events/2023-08_neurips/) |

The OP3 source H5AD can be downloaded separately (approximately 14.2 GB):

```bash
mkdir -p data/raw/op3
curl --fail --location --continue-at - \
  --output data/raw/op3/GSE279945_sc_counts_processed.h5ad \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/GSE279945/suppl/GSE279945_sc_counts_processed.h5ad
```

For sciPlex3, use the matrix and cell/gene annotation files linked under the
GEO sample's supplementary files. The parent series includes other sci-Plex
experiments as well; select sciPlex3. The OP3 competition's
[Kaggle data page](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data)
provides related competition data, but scDVD used the GEO H5AD identified above.

## What is included

| Resource | Availability |
|---|---|
| Exact cell assignments, expected conditions and dose conventions | Included in [`benchmarks/splits/`](../benchmarks/splits/) |
| Ordered 2,000-gene lists and molecular-feature hashes | Included in [`benchmarks/manifests/`](../benchmarks/manifests/) |
| Per-run metrics and table summaries | Included in [`benchmarks/results/`](../benchmarks/results/) |
| scDVD-preprocessed H5AD files | Public download link not yet assigned |
| Exact `unimol.npy` and `drug_lookup.csv` files | Public download links not yet assigned |
| Diffusion and matching mask checkpoints | Public download links not yet assigned |

[`assets.json`](../benchmarks/manifests/assets.json) records feature hashes and
original asset paths. `source_path` fields describe experiment provenance, not
paths available on a fresh clone. Checkpoint provenance is recorded in
[`runs.json`](../benchmarks/manifests/runs.json).

**A fresh clone supports the synthetic example and recorded-table aggregation
after installation. Full paper training and evaluation also require the prepared
assets above.** This release does not provide end-to-end raw-GEO preprocessing
or Uni-Mol feature generation.

## Paper preprocessing conventions

Both datasets use a fixed, ordered **2,000-HVG** space, selected before benchmark
splitting. This summarizes the recorded preparation; independently prepared
H5ADs are not necessarily equivalent.

| Step | sciPlex3 | OP3 |
|---|---|---|
| Expression | GEO matrix, `log1p` transformation | Source H5AD `raw.X` counts, normalize to 10,000 per cell, then `log1p` |
| Gene space | Fixed 2,000 HVGs | Fixed 2,000 HVGs, Scanpy `seurat` flavor |
| Cell context | A549, K562, MCF7 | Six original annotations from `cell_type_orig` |
| Control reference | Vehicle cells | DMSO only (`sm_name == 'Dimethyl Sulfoxide'`) |
| Batch context | Prepared `batch` column | `donor_id` mapped to `batch` |
| Final OOD design | Nine fixed drugs in all repeats | Seven drugs per repeat, changing between repeats |

For OP3, the source `control` column includes positive controls as well as DMSO.
scDVD treats Dabrafenib and Belinostat as drug treatments. Copying the source
`control` or reannotated four-class `cell_type` column unchanged would produce a
different benchmark. The prepared `condition` uses `sm_name`, with DMSO renamed
to `ctrl`; doses come from `dose_uM`, with control dose set to zero.

Runtime scaling by the outer-training expression maximum is separate from this
preprocessing. Do not pass raw counts to `prepare_data.py` or normalize an already
prepared H5AD a second time.

## Molecular features

scDVD consumes fixed [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) molecular
embeddings. Each row of `unimol.npy` must correspond to `drug_idx` in its lookup
table. Newly generated embeddings are not guaranteed to recover the exact paper
matrix; use the recorded hashes to identify paper features.

## Prepare one exact paper split

```bash
python scripts/prepare_data.py --dataset sciplex3 \
  --h5ad /path/to/preprocessed_all_cells.h5ad \
  --features /path/to/original_unimol.npy --lookup /path/to/original_drug_lookup.csv \
  --repeat 1 --split 1 --output data
```

The command checks cell identities, condition/cell-type/control/dose labels,
gene order and molecular-feature hashes, then applies the saved per-cell split.
It preserves expression values and source cell order. Rewriting H5AD may change
its file checksum even when its contents are equivalent. The saved
`split_sha256` is an original-run provenance field, not a hash of the rewritten
H5AD file. Use the original run H5AD for byte-identical artifacts.

## H5AD contract

`X` is a cells × genes dense or CSR expression matrix; `var` supplies ordered
gene names and optionally `id`. Expected `obs` columns are configurable in
`configs/dataset/`:

| Meaning | sciPlex3 | OP3 |
|---|---|---|
| Partition | `unlasting_split` | `unlasting_split` |
| Vehicle indicator | `control` (0/1) | `control` (0/1 or bool) |
| Cell context | `cell_type` | `cell_type` |
| Batch/donor context | `batch` | `batch` |
| Drug | `condition` | `condition` |
| Dose input | `dose / 1000` | `dose_val` |

Additional time/index requirements are in the dataset YAML. The drug lookup
contains `drug_name,drug_idx` and maps into the rows of `unimol.npy`. Every
treated drug must map to a feature row. Do not reorder either resource.
The sciPlex3 model-dose and baseline `dose_val` conventions differ; consult the
saved `dose_conventions.json` rather than matching conditions by raw dose alone.

## Move checkpoints to another machine

The fixed molecular feature buffer is nonpersistent. Checkpoints therefore need
the corresponding feature matrix at inference time:

```python
from scdvd.checkpoints import load_diffusion_checkpoint

model = load_diffusion_checkpoint(
    "diffusion.ckpt", map_location="cpu",
    chemical_feature_path="data/sciplex3/features/unimol.npy",
)
```

The evaluator supplies `chemical_feature_path=${model.chemical_feature_path}`
to both diffusion and mask loaders. Parameter keys load strictly; only unused
population-gate keys in explicitly gate-disabled legacy checkpoints are removed
with a warning. A genuinely gated checkpoint keeps its gate and is historical.
Only load checkpoints whose source you trust.

The evaluator checks the final set4, gate-free, cap-free energy configuration by
default. Historical evaluation requires both `evaluation.require_paper_model=false`
and a distinct label, for example `evaluation.model_label=scDVD_set16_historical`.
It cannot silently label an incompatible checkpoint as the final paper model.

Evaluation also needs the matching training controls, train-derived scaling,
drug/category dictionaries, mask checkpoint and training treated cells for the
empirical support library. A diffusion checkpoint alone is insufficient to
reproduce final masked predictions. New training checkpoints record gene and
category order. Older checkpoints may not contain complete category metadata;
use their exact matching dataset and split. Supplying different molecular
features can change predictions even when dimensions agree.
