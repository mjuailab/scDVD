from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from scdvd.data import UnpairedH5ADStore
from scdvd.checkpoints import load_diffusion_checkpoint, load_mask_checkpoint, validate_dataset_metadata, is_paper_model
from scdvd.runtime import record_run
from scdvd.evaluation import METRIC_NAMES, metric_triplet
from scdvd.masking import EmpiricalSupportLibrary


def _matched_control_moments(
    store: UnpairedH5ADStore,
    cell_idx: int,
    batch_indices: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pool training-control moments for the target condition's contexts."""
    count = 0
    total = np.zeros(store.metadata.num_genes, dtype=np.float64)
    total_squares = np.zeros_like(total)
    for batch_idx in np.unique(batch_indices):
        context = (int(cell_idx), int(batch_idx))
        control_indices = store.control_pools[context]
        context_count = len(control_indices)
        # Match DEPARTURES exactly: calculate each batch's population moments
        # directly with NumPy, then combine batches below. Do not reuse the
        # chunk-accumulated training moments here because round-off at a DE40
        # boundary can otherwise select a different tied gene.
        values = store.expression.rows(control_indices).astype(
            np.float64, copy=False
        )
        mean = values.mean(axis=0)
        variance = values.var(axis=0)
        count += context_count
        total += context_count * mean
        total_squares += context_count * (variance + np.square(mean))
    pooled_mean = total / count
    pooled_population_variance = np.maximum(
        total_squares / count - np.square(pooled_mean), 0.0
    )
    pooled_sample_variance = (
        pooled_population_variance * count / (count - 1)
        if count > 1
        else pooled_population_variance
    )
    return count, pooled_mean, pooled_sample_variance


def _rank_condition_de_genes(
    store: UnpairedH5ADStore,
    target_indices: np.ndarray,
    target_expression: np.ndarray,
) -> np.ndarray:
    """Rank measured genes directly from matched treated/control cells.

    The ranking is condition-specific (cell type, perturbation and dose). It
    uses the absolute Welch t statistic between the real treated cells and
    training controls from the same cell type and observed batches.
    """
    target_count = len(target_expression)
    if target_count < 2:
        raise ValueError("At least two treated cells are required to rank DE genes")
    cell_indices = np.unique(store.cell_idx[target_indices])
    if len(cell_indices) != 1:
        raise ValueError("A DE evaluation group must contain exactly one cell type")
    control_count, control_mean, control_variance = _matched_control_moments(
        store,
        int(cell_indices[0]),
        store.batch_idx[target_indices],
    )
    target = target_expression.astype(np.float64, copy=False)
    target_mean = target.mean(axis=0)
    target_variance = target.var(axis=0, ddof=1)
    standard_error = np.sqrt(
        target_variance / target_count + control_variance / control_count
    )
    scores = np.divide(
        np.abs(target_mean - control_mean),
        standard_error,
        out=np.zeros_like(target_mean),
        where=standard_error > 0,
    )
    scores[~np.isfinite(scores)] = -np.inf
    return np.argsort(-scores, kind="stable")


def _condition_groups(store: UnpairedH5ADStore, config) -> list[tuple[tuple, np.ndarray]]:
    mask = (store.control == 0) & (store.split == str(config.split))
    if config.cell_type is not None:
        cell_idx = store.cell_types.index(str(config.cell_type))
        mask &= store.cell_idx == cell_idx
    if config.drug is not None:
        local_drug_idx = store.drugs.index(str(config.drug))
        mask &= store.local_drug_idx == local_drug_idx

    groups: dict[tuple, list[int]] = {}
    for index in np.where(mask)[0]:
        # Replicate/batch is a model covariate, not a perturbation condition.
        # Pool all replicates for the cell type-drug-dose evaluation unit.
        key = (
            int(store.cell_idx[index]),
            int(store.local_drug_idx[index]),
            float(store.dosage[index]),
            float(store.treatment_time[index]),
            int(store.pert_idx[index]),
        )
        groups.setdefault(key, []).append(int(index))
    minimum = int(config.min_real_cells)
    ordered = [
        (key, np.asarray(indices, dtype=np.int64))
        for key, indices in sorted(groups.items())
        if len(indices) >= minimum
    ]
    if config.max_conditions is not None:
        ordered = ordered[: int(config.max_conditions)]
    if not ordered:
        raise ValueError("No treated conditions match the evaluation filters")
    return ordered


def _make_generation_batch(
    store: UnpairedH5ADStore,
    key: tuple,
    target_indices: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    (
        cell_idx,
        local_drug_idx,
        dosage,
        treatment_time,
        pert_idx,
    ) = key

    # Allocate generated cells to replicates in the same proportion as the
    # pooled real target population. Largest-remainder allocation keeps the
    # total exactly equal to `count` and is deterministic before shuffling.
    real_batches, real_counts = np.unique(
        store.batch_idx[target_indices], return_counts=True
    )
    expected = count * real_counts.astype(np.float64) / real_counts.sum()
    allocated = np.floor(expected).astype(np.int64)
    remainder = count - int(allocated.sum())
    if remainder:
        order = np.argsort(-(expected - allocated), kind="stable")
        allocated[order[:remainder]] += 1
    generated_batches = np.concatenate(
        [
            np.full(batch_count, batch_idx, dtype=np.int64)
            for batch_idx, batch_count in zip(real_batches, allocated)
        ]
    )
    rng.shuffle(generated_batches)

    selected_controls = np.empty(count, dtype=np.int64)
    control_mean = np.empty(
        (count, store.metadata.num_genes), dtype=np.float32
    )
    control_std = np.empty_like(control_mean)
    for batch_idx in real_batches:
        positions = np.where(generated_batches == batch_idx)[0]
        context = (cell_idx, int(batch_idx))
        pool = store.control_pools[context]
        selected_controls[positions] = rng.choice(
            pool, size=len(positions), replace=len(pool) < len(positions)
        )
        control_mean[positions] = store.control_mean[context]
        control_std[positions] = store.control_std[context]

    batch = {
        "source": torch.from_numpy(store.expression.rows(selected_controls)),
        "control_mean": torch.from_numpy(control_mean),
        "control_std": torch.from_numpy(control_std),
        "cell_idx": torch.full((count,), cell_idx, dtype=torch.long),
        "batch_idx": torch.from_numpy(generated_batches),
        "local_drug_idx": torch.full((count,), local_drug_idx, dtype=torch.long),
        "pert_idx": torch.full((count,), pert_idx, dtype=torch.long),
        "dosage": torch.full((count,), dosage, dtype=torch.float32),
        "dose_idx": torch.from_numpy(store.dose_idx[target_indices]).mode().values.repeat(count),
        "time_idx": torch.from_numpy(store.time_idx[target_indices]).mode().values.repeat(count),
        "treatment_time": torch.full(
            (count,), treatment_time, dtype=torch.float32
        ),
    }
    return {name: value.to(device) for name, value in batch.items()}


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


#: `bg` is every gene outside the top-40 DE ranking. Comparing it against
#: `de20`/`de40` localizes a metric gap to the perturbation signal or to the
#: ~1960 background genes that dominate the all-gene energy distance.
SCOPES = ("all", "de20", "de40", "bg")


def _summary(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for scope in SCOPES:
        for metric in METRIC_NAMES:
            key = f"{scope}_{metric}"
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            output[key] = {
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "std": float(np.std(finite)) if len(finite) else float("nan"),
                "conditions": int(len(finite)),
            }
    return output


def _print_table(summary: dict[str, dict[str, float | int]]) -> None:
    print("\nScope\tRMSE (mean±std)\tE-distance (mean±std)\tEMD (mean±std)")
    for scope in SCOPES:
        cells = []
        for metric in ("rmse", "e_distance", "emd"):
            value = summary[f"{scope}_{metric}"]
            cells.append(f"{value['mean']:.4f}±{value['std']:.4f}")
        print(f"{scope.upper()}\t" + "\t".join(cells))

    print(
        "\nEnergy-distance decomposition  "
        "ED^2 = 2*cross - within_real - within_gen"
    )
    print("Scope\tcross\twithin_real\twithin_gen\tdispersion (gen/real)")
    for scope in SCOPES:
        cells = [
            f"{summary[f'{scope}_{metric}']['mean']:.4f}"
            for metric in ("ed_cross", "ed_within_real", "ed_within_gen")
        ]
        ratio = summary[f"{scope}_ed_dispersion_ratio"]
        cells.append(f"{ratio['mean']:.4f}±{ratio['std']:.4f}")
        print(f"{scope.upper()}\t" + "\t".join(cells))


@hydra.main(version_base=None, config_path="configs", config_name="sciplex3")
def main(config: DictConfig) -> None:
    if config.checkpoint is None:
        raise ValueError("Set checkpoint=/path/to/last.ckpt")
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    rng = np.random.default_rng(int(config.seed))
    # The empirical mask draws from its own stream so that toggling
    # apply_empirical_mask does not shift the control cells and real-cell
    # orderings drawn for every later condition. Mask-on and mask-off runs are
    # then paired condition by condition instead of differing by sampling noise.
    mask_rng = np.random.default_rng(int(config.seed) + 1)
    real_rng = np.random.default_rng(int(config.seed) + 2)
    num_real_cells = config.evaluation.get("num_real_cells", None)
    match_real_count = bool(config.evaluation.get("match_real_count", False))
    if match_real_count and (
        num_real_cells is not None or config.evaluation.max_real_cells is not None
    ):
        raise ValueError("match_real_count requires all real cells without resampling or capping")
    if num_real_cells is not None and int(num_real_cells) < 2:
        raise ValueError("evaluation.num_real_cells must be at least 2")
    if int(config.evaluation.num_generated) < 2:
        raise ValueError("evaluation.num_generated must be at least 2")
    if (
        config.evaluation.max_real_cells is not None
        and int(config.evaluation.max_real_cells) < 2
    ):
        raise ValueError("evaluation.max_real_cells must be at least 2")

    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    store = UnpairedH5ADStore(**dataset_config)
    groups = _condition_groups(store, config.evaluation)
    if not groups:
        raise ValueError("No eligible evaluation conditions; check split, filters and min_real_cells")

    requested_device = str(config.evaluation.device)
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    model = load_diffusion_checkpoint(
        config.checkpoint, map_location=device,
        chemical_feature_path=config.get("chemical_feature_path", None),
    ).to(device)
    validate_dataset_metadata(model, store.metadata)
    model_label = str(config.evaluation.get("model_label", "scDVD"))
    if not is_paper_model(model):
        if bool(config.evaluation.get("require_paper_model", True)):
            raise ValueError("Checkpoint is not the final set4 scDVD configuration; do not reuse it as a paper prediction")
        if model_label.lower() == "scdvd":
            raise ValueError("Historical experiments require an explicit distinct evaluation.model_label")
    model.eval()
    # Classifier-free guidance is a checkpoint hyperparameter, but it is a
    # pure inference-time knob: overriding it here isolates how much of a
    # dispersion gap comes from guidance sharpening the conditional mode.
    guidance_override = config.evaluation.get("guidance_scale", None)
    if guidance_override is not None:
        print(
            f"guidance_scale override: {model.guidance_scale} "
            f"-> {float(guidance_override)}"
        )
        model.guidance_scale = float(guidance_override)
    mask_model = None
    support_library = None
    if bool(config.evaluation.apply_empirical_mask):
        if config.mask_checkpoint is None:
            raise ValueError(
                "Set mask_checkpoint=... when evaluation.apply_empirical_mask=true"
            )
        mask_model = load_mask_checkpoint(
            config.mask_checkpoint, map_location=device,
            chemical_feature_path=config.get("chemical_feature_path", None),
        ).to(device)
        validate_dataset_metadata(mask_model, store.metadata)
        mask_model.eval()
        support_library = EmpiricalSupportLibrary(store, split="train")

    record_run(config.evaluation.output_dir, config, stage="evaluate", metadata=store.metadata,
               artifacts={"diffusion_checkpoint": config.checkpoint,
                          "mask_checkpoint": config.mask_checkpoint if mask_model is not None else None,
                          "chemical_features": model.hparams["model_config"]["chemical_feature_path"]})

    rows: list[dict] = []
    for position, (key, target_indices) in enumerate(groups, start=1):
        (
            cell_idx,
            local_drug_idx,
            dosage,
            _,
            _,
        ) = key
        condition_real = store.expression.rows(target_indices)
        real_count = len(condition_real)
        if config.evaluation.max_real_cells is not None:
            real_count = min(real_count, int(config.evaluation.max_real_cells))
        selected_positions = rng.choice(
            len(condition_real), size=real_count, replace=False
        )
        # Preserve the generation RNG stream when changing only the real
        # population used for evaluation. Small populations use replacement.
        if num_real_cells is not None:
            real_count = int(num_real_cells)
            selected_positions = real_rng.choice(
                len(condition_real), size=real_count,
                replace=len(condition_real) < real_count,
            )
        real = condition_real[selected_positions] * store.normalization_scale
        batch = _make_generation_batch(
            store,
            key,
            target_indices,
            count=real_count if match_real_count else int(config.evaluation.num_generated),
            rng=rng,
            device=device,
        )
        with torch.no_grad():
            generated = model.bridge_sample(
                batch,
                ddim_steps=int(config.evaluation.ddim_steps),
                clip_absolute_each_step=bool(
                    config.evaluation.get("clip_absolute_each_step", False)
                ),
                latent_source=str(
                    config.evaluation.get("latent_source", "noise")
                ),
            )
            if mask_model is not None:
                probabilities = (
                    mask_model.predict_probabilities(batch)
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                empirical_mask = support_library.sample(
                    probabilities,
                    count=len(generated),
                    rng=mask_rng,
                    topk_conditions=int(
                        config.evaluation.mask_topk_conditions
                    ),
                    feature_topk=int(config.evaluation.mask_feature_topk),
                    high=float(config.evaluation.mask_high_threshold),
                    low=float(config.evaluation.mask_low_threshold),
                )
                generated = generated * torch.from_numpy(empirical_mask).to(
                    generated.device
                )
        generated = generated.cpu().numpy() * store.normalization_scale
        if match_real_count:
            assert len(generated) == len(condition_real) == real_count

        cell_type = store.cell_types[cell_idx]
        drug = store.drugs[local_drug_idx]
        row = {
            "cell_type": cell_type,
            "batch": "pooled",
            "drug": drug,
            "dosage": dosage,
            "real_cells": real_count,
            "generated_cells": len(generated),
        }
        if bool(config.evaluation.get("save_predictions", False)):
            prediction_dir = Path(config.evaluation.output_dir) / "predictions"
            prediction_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_dir / f"condition_{position:04d}.npz",
                GT=real, **{model_label: generated},
                gene_names=np.asarray(store.metadata.gene_names, dtype=str),
                cell_type=np.asarray(cell_type), drug=np.asarray(drug), dosage=dosage,
            )
        ranked = _rank_condition_de_genes(
            store,
            target_indices,
            condition_real,
        )
        row.update(_prefixed(metric_triplet(real, generated), "all"))
        row.update(_prefixed(metric_triplet(real, generated, ranked[:20]), "de20"))
        row.update(_prefixed(metric_triplet(real, generated, ranked[:40]), "de40"))
        row.update(_prefixed(metric_triplet(real, generated, ranked[40:]), "bg"))
        rows.append(row)
        if match_real_count:
            progress_path = Path(config.evaluation.output_dir) / "condition_metrics.partial.csv"
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("w" if position == 1 else "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if position == 1:
                    writer.writeheader()
                writer.writerow(row)
        print(
            f"[{position}/{len(groups)}] {cell_type}/{drug}/{dosage:g} "
            f"real={real_count} generated={len(generated)} "
            f"RMSE={row['all_rmse']:.4f} ED={row['all_e_distance']:.4f} "
            f"disp={row['all_ed_dispersion_ratio']:.4f}"
        )

    summary = _summary(rows)
    output_dir = Path(config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions_path = output_dir / "condition_metrics.csv"
    with conditions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    summary_csv_path = output_dir / "summary.csv"
    flat_summary = {"model": model_label}
    for key, value in summary.items():
        flat_summary[key] = value["mean"]
        flat_summary[f"{key}_std"] = value["std"]
    with summary_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_summary))
        writer.writeheader()
        writer.writerow(flat_summary)
    _print_table(summary)
    print(f"\nSaved condition metrics to {conditions_path.resolve()}")
    print(f"Saved aggregate metrics to {summary_path.resolve()}")
    print(f"Saved paper-table row to {summary_csv_path.resolve()}")


if __name__ == "__main__":
    main()
