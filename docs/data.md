# Data and checkpoint assets

`paper/manifests/assets.json` identifies the feature matrices and lookup tables
by SHA-256. Download URLs are currently unassigned. The preprocessed H5ADs,
features and checkpoints are not embedded in the source repository.

The manuscript reports sciPlex3 log1p expression and OP3 normalization to 10,000
counts per cell followed by log1p, with a fixed 2,000-HVG gene space. This release
starts from those prepared representations; do not run raw counts through the
prepared-data command and assume they reproduce the paper.

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
H5AD file. Use the original released H5AD for byte-identical artifacts.

## H5AD contract

`X` is a cells × genes dense or CSR expression matrix; `var` supplies ordered
gene names and optionally `id`. Expected `obs` columns are configurable in
`src/scdvd/configs/dataset/`:

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
