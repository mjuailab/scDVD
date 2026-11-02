# Reproduction protocol

## Final experiments

| Dataset | Cohort | Final experiment | Repeats × splits |
|---|---|---|---|
| sciPlex3 | fixed9 | `sciplex3_set4_fixed_random_3x5_20261027` | 3 × 5 |
| OP3 | repeat7 | `OP3_set4_repeat3_fold5_20261031` | 3 × 5 |

All 30 selected jobs have completed train/TEST/OOD records. sciPlex3's nine
OOD drugs stay fixed across repeats; OP3's seven OOD drugs change by repeat.
`paper/splits/` contains original cell assignments, dose conventions, expected
conditions and split manifests. The separate sciPlex3 repeat9 cohort is excluded
from the final paper table bundle.

The final diffusion runs use set size 4 but retain the cell membership of the
reference size-16 internal partition. `training.partition_set_size=16` first
determines membership; training and validation cells are then regrouped into
size-four sets. This does **not** make the denoiser set size 16. Changing it to
4 changes validation membership and no longer reproduces the final experiment.

Training uses seed 42, eight sets per microbatch, four accumulated microbatches,
100 maximum epochs, early stopping patience 5 and minimum validation loss for
checkpoint selection. Evaluation also uses seed 42. The final set4 runs reused
the already-trained mask checkpoint from the same parent split. `train-mask`
provides the independent training procedure; exact historical inference should
use the recorded matching mask checkpoint, not an arbitrary newly trained mask.

## Evaluation and tables

Eligible conditions have at least ten GT treated cells. Conditions pool batches
for evaluation and use the same generated cell count as GT. DE20 and DE40 use
the absolute Welch t-statistic comparing GT treated expression with matched
training controls, exclusively for evaluation. Metrics are reported in the
processed H5AD expression units after undoing model scaling, not raw UMI counts.

Aggregation is: equal-weight condition mean within each run, mean over five
splits within each repeat, then mean and **sample SD** over three repeat means.
`scripts/aggregate_results.py` rejects duplicate, missing and nonfinite runs.
The included run-metric CSV was extracted from final set4 comparison reports;
its aggregation recreates the scDVD entries, not the baseline entries.

`paper/manifests/runs.json` preserves the original checkpoint hashes and their
verification status. Exporting a recorded hash is not a fresh hash verification.
The local refactor validation report separately identifies the actual checkpoints
rehashed and compared with frozen-source inference.

## Source and checkpoint compatibility

The implementation starts from the frozen SciPlex3 final-run source; its core
Python files match those of the final OP3 source. Compared with the old live
folder, the final source removes the disabled population-gate parameters and
preserves reference split membership when regrouping size-four sets.

Loss functions are now in `scdvd.losses` and remain re-exported from
`scdvd.module`. Checkpoint class paths and parameter names remain stable.
Entry-point code moved into the installable package, with root-level wrappers.

```bash
python scripts/verify_frozen_compatibility.py \
  --reference /path/to/final_run/source \
  --checkpoint /path/to/final_run/checkpoint.ckpt \
  --output outputs/compatibility.json
```

This CPU check compares parameter keys/values, fresh seeded initialization and
four-step DDIM outputs against the frozen implementation. It does not substitute
for rerunning the complete 50-step benchmark on every condition.

## Figures and analyses

The manuscript's Figures 2–6 involve additional downstream direction/Pearson,
UMAP, external-signature and GSEA workflows. Figures 7–8 use separate ablation
and sensitivity experiments. This repository refactor includes model training,
benchmark evaluation and table aggregation; it does not claim to reproduce all
figure panels from one command. Preserve each analysis's own run/selection
provenance when porting those workflows. The 35-gene external panel was selected
post hoc, as described in the manuscript, and is not an unbiased ranking test.

For subsequent figure work, use the workspace's `pub_plot_kit` foundation,
label observed data `GT`, keep signed NES scales centered on zero, and distinguish
missing/untestable results from nonsignificant results.
