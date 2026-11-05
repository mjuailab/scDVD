# Running experiments

[Back to README](../README.md) · [Data preparation](data.md)

Run these commands from the repository root after installation. Training and
benchmark evaluation require the prepared H5AD and molecular features described
in the data guide. For a self-contained execution example, use
`python scripts/cpu_demo.py --output outputs/cpu_demo`.

## Configuration

The entry configurations are [`sciplex3.yaml`](../configs/sciplex3.yaml) and
[`op3.yaml`](../configs/op3.yaml). Shared defaults live under `configs/model/`
and `configs/training/`; column mappings live under `configs/dataset/`.
These YAMLs are bundled as `scdvd.configs` on installation.

Inspect a resolved configuration without starting a run:

```bash
scdvd train --config-name sciplex3 --cfg job --resolve
scdvd evaluate --config-name op3 --cfg job --resolve
```

Hydra accepts `key=value` overrides after the command:

| Override | Default | Purpose |
|---|---|---|
| `repeat=1 split=1` | 1, 1 | Prepared repeat/split |
| `paths.data_root=/path/to/data` | `data` | Dataset root |
| `paths.output_root=/path/to/outputs` | `outputs` | Output root |
| `training.accelerator=cpu` | `gpu` | Diffusion training device |
| `mask_training.accelerator=cpu` | `gpu` | Mask training device |
| `evaluation.device=cpu` | `auto` | Evaluation device |
| `evaluation.split=ood` | `test` | Held-out drugs or held-out conditions |
| `evaluation.save_predictions=true` | `false` | Save expression arrays |

You can also set paths for the current shell:

```bash
export SCDVD_DATA_ROOT="$PWD/data"
export SCDVD_OUTPUT_ROOT="$PWD/outputs"
```

Commands keep the working directory unchanged. Hydra writes configuration logs
beneath `outputs/hydra/`; model checkpoints and results use the directories below.

## Train one split

```bash
CUDA_VISIBLE_DEVICES=0 scdvd train --config-name sciplex3 repeat=1 split=1
CUDA_VISIBLE_DEVICES=0 scdvd train-mask --config-name sciplex3 repeat=1 split=1
```

For OP3:

```bash
CUDA_VISIBLE_DEVICES=0 scdvd train --config-name op3 repeat=1 split=1
CUDA_VISIBLE_DEVICES=0 scdvd train-mask --config-name op3 repeat=1 split=1
```

Default training outputs for sciPlex3 r1/s1 are:

```text
outputs/
├── sciplex3_r1_s1_set4/
│   ├── <epoch>-<step>.ckpt      # Best validation diffusion checkpoint
│   ├── last.ckpt
│   ├── resolved_config.yaml
│   └── run_provenance.json
└── sciplex3_r1_s1_mask/
    ├── <epoch>-<step>.ckpt      # Best validation mask checkpoint
    ├── last.ckpt
    ├── resolved_config.yaml
    └── run_provenance.json
```

Use the **best validation** paths printed at the end of training for evaluation.
`last.ckpt` is the final epoch checkpoint and need not be the best one. For a new
attempt, choose another `paths.output_root` to keep outputs separate.

Diffusion defaults are set size 4, 100 maximum epochs, early stopping patience 5,
eight sets per microbatch and four accumulated microbatches. The mask predictor
defaults to two epochs. Final paper runs reused the corresponding parent split's
mask checkpoint; retraining the mask is a new experiment, not a byte-identical
reconstruction of those runs.

For CPU training, append `training.accelerator=cpu training.num_workers=0` to
`scdvd train`, or `mask_training.accelerator=cpu mask_training.num_workers=0` to
`scdvd train-mask`. Full-size paper training on CPU can be slow.

## Evaluate TEST and OOD

Set the checkpoint paths printed by training:

```bash
SCDVD_DIFFUSION_CKPT=/path/to/best_diffusion.ckpt
SCDVD_MASK_CKPT=/path/to/best_mask.ckpt

for evaluation_split in test ood; do
  scdvd evaluate --config-name sciplex3 repeat=1 split=1 \
    checkpoint="$SCDVD_DIFFUSION_CKPT" \
    mask_checkpoint="$SCDVD_MASK_CKPT" \
    evaluation.split="$evaluation_split" \
    evaluation.save_predictions=true
done
```

Use matching data, feature rows, gene order and checkpoints for each dataset,
repeat and split. `evaluation.device=cpu` forces CPU inference;
`CUDA_VISIBLE_DEVICES=0` selects a visible GPU when using the default `auto`.

The evaluator generates as many cells as the observed condition contains and
reports ALL/DE20/DE40 metrics for conditions with at least ten GT cells. Outputs
are in `outputs/<dataset>_r<repeat>_s<split>/test/` and `ood/`. `summary.csv` and
`summary.json` contain per-run aggregates; `condition_metrics.csv` holds individual
conditions. Prediction archives contain `GT`, `scDVD`, ordered genes and
condition metadata in the processed H5AD expression scale.

For a pipeline check on prepared data, add `evaluation.max_conditions=1` and
choose a separate `evaluation.output_dir=outputs/evaluation_check`. Partial
outputs are not complete benchmark results. Changing DDIM steps or sampling
settings also changes the experiment.

## Train all repeats and splits

Prepare all 15 split files first. This loop trains sequentially on one selected GPU:

```bash
(
  set -e
  for repeat in 1 2 3; do
    for split in 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=0 scdvd train --config-name sciplex3 \
        repeat="$repeat" split="$split"
      CUDA_VISIBLE_DEVICES=0 scdvd train-mask --config-name sciplex3 \
        repeat="$repeat" split="$split"
    done
  done
)
```

Evaluate each run using its own best-validation checkpoint pair. The aggregation
script expects a long-form table with columns
`dataset,repeat,split,evaluation_split,genes,metric,value`, where `value` is a
condition-average metric for one run. It does not accept `summary.csv` directly.
The included [run metrics](../benchmarks/results/scdvd_set4_run_metrics.csv)
show the schema. See [Reproduction protocol](reproduction.md) for aggregation.

## Python entry points

Use the installed `scdvd` command or the equivalent Python module entry point:

```bash
python -m scdvd train --config-name sciplex3 --cfg job
python -m scdvd train-mask --config-name sciplex3 --cfg job
python -m scdvd evaluate --config-name sciplex3 --cfg job
```

Install the package before running these commands. Existing imports under
`scdvd.models`, `scdvd.module` and `scdvd.checkpoints` retain their module paths.
