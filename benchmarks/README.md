# Paper benchmarks

[Back to README](../README.md) · [Reproduction protocol](../docs/reproduction.md)

This directory contains the exact split definitions and recorded results for
the final **set4 scDVD** experiments.

| Directory | Contents |
|---|---|
| `splits/<dataset>/r<repeat>/s<split>/` | Cell assignments, expected conditions, dose conventions and split metadata |
| `manifests/` | Run provenance, checkpoint/feature hashes, ordered gene lists and frozen-source hashes |
| `results/` | Per-run metrics and the aggregated scDVD table entries |

The datasets are `sciplex3` (fixed9) and `op3` (repeat7), each with 3 repeats × 5
repeated condition holdouts. These are not disjoint cross-validation folds.
All 30 selected runs have completed train, TEST and OOD records. Set16 outputs
and the separate sciPlex3 repeat9 cohort are not the final results here.

From the repository root:

```bash
python scripts/aggregate_results.py \
  --input benchmarks/results/scdvd_set4_run_metrics.csv \
  --output outputs/scdvd_set4_summary.csv
```

The input contains per-run condition means. The script averages five splits
within each repeat, then reports the mean and sample standard deviation of the
three repeat means. This reproduces recorded scDVD entries without rerunning
inference and does not reproduce the baseline entries.

Paths stored inside provenance manifests identify the original experiments.
They are not download URLs or expected paths on a fresh checkout. Large H5ADs,
feature matrices and model checkpoints are separate assets; see
[Data preparation](../docs/data.md) for their availability.
