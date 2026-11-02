from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .h5ad import H5ExpressionMatrix, read_obs_column


@dataclass(frozen=True)
class DatasetMetadata:
    num_genes: int
    num_cell_types: int
    num_batches: int
    num_local_drugs: int
    cell_types: tuple[str, ...]
    batches: tuple[str, ...]
    drugs: tuple[str, ...]
    gene_names: tuple[str, ...]
    gene_ids: tuple[str, ...]


class UnpairedH5ADStore:
    """Shared metadata and control statistics for unpaired perturbation data."""

    def __init__(
        self,
        *,
        path: str | Path,
        split_key: str,
        train_split: str = "train",
        control_key: str = "control",
        cell_type_key: str = "cell_type",
        batch_key: str = "batch",
        drug_key: str = "condition",
        dose_key: str = "dose_val",
        dose_divisor: float = 1.0,
        dose_idx_key: str | None = "xpert_dose_idx",
        time_key: str | None = "xpert_time",
        time_idx_key: str | None = "xpert_time_idx",
        pert_idx_key: str | None = "xpert_pert_idx",
        pert_mapping_path: str | Path | None = None,
        pert_mapping_drug_column: str = "drug_name",
        pert_mapping_index_column: str = "drug_idx",
        normalization_scale: float | str = 5.0,
        stats_chunk_size: int = 512,
    ) -> None:
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as handle:
            obs = handle["obs"]
            var = handle["var"]
            gene_index_key = var.attrs.get("_index")
            if isinstance(gene_index_key, bytes):
                gene_index_key = gene_index_key.decode("utf-8")
            if not gene_index_key or gene_index_key not in var:
                raise KeyError("H5AD var dataframe does not define a readable index")
            gene_names = read_obs_column(var, str(gene_index_key)).astype(str)
            gene_ids = (
                read_obs_column(var, "id").astype(str)
                if "id" in var
                else gene_names.copy()
            )
            self.split = read_obs_column(obs, split_key).astype(str)
            self.control = read_obs_column(obs, control_key).astype(np.int64)
            cell_names = read_obs_column(obs, cell_type_key).astype(str)
            batch_names = read_obs_column(obs, batch_key).astype(str)
            drug_names = read_obs_column(obs, drug_key).astype(str)
            dose_divisor = float(dose_divisor)
            if not np.isfinite(dose_divisor) or dose_divisor <= 0:
                raise ValueError("dose_divisor must be a positive finite number")
            self.dosage = (
                read_obs_column(obs, dose_key).astype(np.float32)
                / dose_divisor
            )
            self.treatment_time = (
                np.zeros_like(self.dosage, dtype=np.float32)
                if time_key is None
                else read_obs_column(obs, time_key).astype(np.float32)
            )
            if dose_idx_key is None:
                _, self.dose_idx = np.unique(self.dosage, return_inverse=True)
                self.dose_idx = self.dose_idx.astype(np.int64)
            else:
                self.dose_idx = read_obs_column(obs, dose_idx_key).astype(np.int64)
            if time_idx_key is None:
                _, self.time_idx = np.unique(
                    self.treatment_time, return_inverse=True
                )
                self.time_idx = self.time_idx.astype(np.int64)
            else:
                self.time_idx = read_obs_column(obs, time_idx_key).astype(np.int64)
            if pert_idx_key is None:
                self.pert_idx = np.full(len(drug_names), -1, dtype=np.int64)
            else:
                self.pert_idx = read_obs_column(obs, pert_idx_key).astype(np.int64)

        if pert_mapping_path is not None:
            mapping_path = Path(pert_mapping_path).expanduser().resolve()
            if not mapping_path.is_file():
                raise FileNotFoundError(mapping_path)
            with mapping_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {pert_mapping_drug_column, pert_mapping_index_column}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        f"Perturbation mapping must contain columns {sorted(required)}"
                    )
                perturbation_lookup = {
                    row[pert_mapping_drug_column]: int(row[pert_mapping_index_column])
                    for row in reader
                }
            remapped = np.asarray(
                [perturbation_lookup.get(name, -1) for name in drug_names],
                dtype=np.int64,
            )
            treated_missing = (self.control == 0) & (remapped < 0)
            if treated_missing.any():
                missing_names = sorted(set(drug_names[treated_missing].tolist()))
                raise ValueError(
                    "Treated drugs missing from perturbation mapping: "
                    f"{missing_names[:20]}"
                )
            self.pert_idx = remapped

        train_mask = self.split == train_split
        if isinstance(normalization_scale, str):
            if normalization_scale != "train_max":
                raise ValueError(
                    "normalization_scale must be a positive number or 'train_max'"
                )
            raw_expression = H5ExpressionMatrix(
                self.path, scale=1.0, clip_upper=False
            )
            resolved_scale = raw_expression.maximum(
                np.where(train_mask)[0], chunk_size=stats_chunk_size
            )
            raw_expression.close()
            if resolved_scale <= 0:
                raise ValueError("Training expression maximum must be positive")
        else:
            resolved_scale = float(normalization_scale)
        self.normalization_scale = resolved_scale
        self.expression = H5ExpressionMatrix(self.path, scale=resolved_scale)

        self.cell_types = tuple(sorted(np.unique(cell_names)))
        self.batches = tuple(sorted(np.unique(batch_names)))
        self.drugs = tuple(sorted(np.unique(drug_names)))
        cell_lookup = {name: index for index, name in enumerate(self.cell_types)}
        batch_lookup = {name: index for index, name in enumerate(self.batches)}
        drug_lookup = {name: index for index, name in enumerate(self.drugs)}
        self.cell_idx = np.asarray([cell_lookup[name] for name in cell_names], dtype=np.int64)
        self.batch_idx = np.asarray([batch_lookup[name] for name in batch_names], dtype=np.int64)
        self.local_drug_idx = np.asarray([drug_lookup[name] for name in drug_names], dtype=np.int64)

        self.metadata = DatasetMetadata(
            num_genes=self.expression.shape[1],
            num_cell_types=len(self.cell_types),
            num_batches=len(self.batches),
            num_local_drugs=len(self.drugs),
            cell_types=self.cell_types,
            batches=self.batches,
            drugs=self.drugs,
            gene_names=tuple(gene_names.tolist()),
            gene_ids=tuple(gene_ids.tolist()),
        )

        control_mask = train_mask & (self.control == 1)
        self.control_pools: dict[tuple[int, int], np.ndarray] = {}
        self.control_mean: dict[tuple[int, int], np.ndarray] = {}
        self.control_std: dict[tuple[int, int], np.ndarray] = {}
        # Float64, unfloored moments used only for fair DE-gene ranking.
        # Training keeps its stabilized float32 mean/std below.
        self.control_de_mean: dict[tuple[int, int], np.ndarray] = {}
        self.control_de_variance: dict[tuple[int, int], np.ndarray] = {}

        contexts = sorted(set(zip(self.cell_idx[control_mask], self.batch_idx[control_mask])))
        for context in contexts:
            indices = np.where(
                control_mask
                & (self.cell_idx == context[0])
                & (self.batch_idx == context[1])
            )[0]
            self.control_pools[context] = indices
            sum_values = np.zeros(self.metadata.num_genes, dtype=np.float64)
            sum_squares = np.zeros(self.metadata.num_genes, dtype=np.float64)
            for start in range(0, len(indices), stats_chunk_size):
                values = self.expression.rows(indices[start : start + stats_chunk_size])
                sum_values += values.sum(axis=0, dtype=np.float64)
                sum_squares += np.square(values, dtype=np.float64).sum(axis=0)
            mean = sum_values / len(indices)
            de_variance = np.maximum(
                sum_squares / len(indices) - np.square(mean), 0.0
            )
            self.control_de_mean[context] = mean.copy()
            self.control_de_variance[context] = de_variance
            variance = np.maximum(de_variance, 1e-6)
            self.control_mean[context] = mean.astype(np.float32)
            self.control_std[context] = np.sqrt(variance).astype(np.float32)

        treated_contexts = set(zip(self.cell_idx[self.control == 0], self.batch_idx[self.control == 0]))
        missing = treated_contexts.difference(self.control_pools)
        if missing:
            raise ValueError(f"No training controls are available for contexts: {sorted(missing)}")

    def dataset(
        self,
        split: str,
        *,
        include_controls_as_targets: bool = False,
        max_samples: int | None = None,
    ) -> "UnpairedPerturbationDataset":
        return UnpairedPerturbationDataset(
            self,
            split,
            include_controls_as_targets=include_controls_as_targets,
            max_samples=max_samples,
        )

    def mixed_dataset(
        self,
        split: str,
        *,
        max_samples: int | None = None,
    ) -> "MixedPerturbationDataset":
        return MixedPerturbationDataset(self, split, max_samples=max_samples)

    def population_dataset(
        self,
        split: str,
        *,
        set_size: int,
        max_samples: int | None = None,
        seed: int = 1337,
    ) -> "PopulationPerturbationDataset":
        return PopulationPerturbationDataset(
            self,
            split,
            set_size=set_size,
            max_samples=max_samples,
            seed=seed,
        )


class UnpairedPerturbationDataset(Dataset):
    """Treated targets plus an independently sampled source control cell."""

    def __init__(
        self,
        store: UnpairedH5ADStore,
        split: str,
        *,
        include_controls_as_targets: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.store = store
        mask = store.split == split
        if not include_controls_as_targets:
            mask &= store.control == 0
        self.indices = np.where(mask)[0]
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]
        if not len(self.indices):
            raise ValueError(f"No samples found for split='{split}'")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_index = int(self.indices[item])
        context = (
            int(self.store.cell_idx[target_index]),
            int(self.store.batch_idx[target_index]),
        )
        pool = self.store.control_pools[context]
        source_index = int(pool[torch.randint(len(pool), size=()).item()])
        return {
            "target": torch.from_numpy(self.store.expression.row(target_index)),
            "source": torch.from_numpy(self.store.expression.row(source_index)),
            "control_mean": torch.from_numpy(self.store.control_mean[context]),
            "control_std": torch.from_numpy(self.store.control_std[context]),
            "cell_idx": torch.tensor(context[0], dtype=torch.long),
            "batch_idx": torch.tensor(context[1], dtype=torch.long),
            "local_drug_idx": torch.tensor(
                self.store.local_drug_idx[target_index], dtype=torch.long
            ),
            "pert_idx": torch.tensor(self.store.pert_idx[target_index], dtype=torch.long),
            "dosage": torch.tensor(self.store.dosage[target_index], dtype=torch.float32),
            "dose_idx": torch.tensor(self.store.dose_idx[target_index], dtype=torch.long),
            "time_idx": torch.tensor(self.store.time_idx[target_index], dtype=torch.long),
            "treatment_time": torch.tensor(
                self.store.treatment_time[target_index], dtype=torch.float32
            ),
        }


class MixedPerturbationDataset(Dataset):
    """Natural control/treated rows for Doloris-style branch routing.

    Unlike :class:`UnpairedPerturbationDataset`, this dataset does not attach an
    equally weighted source cell to every treated cell. Controls appear with
    their natural frequency in the requested split.
    """

    def __init__(
        self,
        store: UnpairedH5ADStore,
        split: str,
        *,
        max_samples: int | None = None,
    ) -> None:
        self.store = store
        self.indices = np.where(store.split == split)[0]
        if max_samples is not None:
            self.indices = self.indices[: int(max_samples)]
        if not len(self.indices):
            raise ValueError(f"No samples found for split='{split}'")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        context = (
            int(self.store.cell_idx[index]),
            int(self.store.batch_idx[index]),
        )
        return {
            "expression": torch.from_numpy(self.store.expression.row(index)),
            "is_control": torch.tensor(
                bool(self.store.control[index] == 1), dtype=torch.bool
            ),
            "control_mean": torch.from_numpy(self.store.control_mean[context]),
            "control_std": torch.from_numpy(self.store.control_std[context]),
            "cell_idx": torch.tensor(context[0], dtype=torch.long),
            "batch_idx": torch.tensor(context[1], dtype=torch.long),
            "local_drug_idx": torch.tensor(
                self.store.local_drug_idx[index], dtype=torch.long
            ),
            "pert_idx": torch.tensor(self.store.pert_idx[index], dtype=torch.long),
            "dosage": torch.tensor(self.store.dosage[index], dtype=torch.float32),
            "dose_idx": torch.tensor(self.store.dose_idx[index], dtype=torch.long),
            "time_idx": torch.tensor(self.store.time_idx[index], dtype=torch.long),
            "treatment_time": torch.tensor(
                self.store.treatment_time[index], dtype=torch.float32
            ),
        }


class PopulationPerturbationDataset(Dataset):
    """Condition-homogeneous treated sets with matched clean control sets."""

    def __init__(
        self,
        store: UnpairedH5ADStore,
        split: str,
        *,
        set_size: int,
        max_samples: int | None = None,
        seed: int = 1337,
    ) -> None:
        self.store = store
        self.set_size = int(set_size)
        if self.set_size < 2:
            raise ValueError("population set_size must be at least 2")
        treated = np.where((store.split == split) & (store.control == 0))[0]
        groups: dict[tuple, list[int]] = {}
        for index in treated:
            key = (
                int(store.cell_idx[index]),
                int(store.batch_idx[index]),
                int(store.local_drug_idx[index]),
                int(store.pert_idx[index]),
                float(store.dosage[index]),
                float(store.treatment_time[index]),
                int(store.dose_idx[index]),
                int(store.time_idx[index]),
            )
            groups.setdefault(key, []).append(int(index))

        self._regroup_pool = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in groups.items()
        }
        self.sets = self._make_sets(seed)
        if max_samples is not None:
            maximum_sets = max(1, int(max_samples) // self.set_size)
            self.sets = self.sets[:maximum_sets]
        if not self.sets:
            raise ValueError(f"No treated population sets found for split={split!r}")

    def _make_sets(self, seed: int) -> list[np.ndarray]:
        """Repartition each condition pool deterministically for one epoch."""
        rng = np.random.default_rng(int(seed))
        sets: list[np.ndarray] = []
        for key in sorted(self._regroup_pool):
            indices = rng.permutation(self._regroup_pool[key])
            for start in range(0, len(indices), self.set_size):
                chunk = indices[start : start + self.set_size]
                if len(chunk) < self.set_size:
                    padding = rng.choice(
                        chunk,
                        size=self.set_size - len(chunk),
                        replace=True,
                    )
                    chunk = np.concatenate([chunk, padding])
                sets.append(chunk)
        return sets

    def restrict_regroup_pool_to_sets(self, sets: list[np.ndarray]) -> None:
        """Keep regrouping inside a pre-existing train/validation partition."""
        retained = {
            int(index)
            for population_set in sets
            for index in np.asarray(population_set).tolist()
        }
        self._regroup_pool = {
            key: np.asarray(
                [index for index in indices if int(index) in retained],
                dtype=np.int64,
            )
            for key, indices in self._regroup_pool.items()
        }
        self._regroup_pool = {
            key: indices
            for key, indices in self._regroup_pool.items()
            if len(indices)
        }
        self.sets = [np.asarray(values, dtype=np.int64) for values in sets]

    def regroup(self, seed: int) -> None:
        """Create new condition-homogeneous cell sets without changing the pool."""
        self.sets = self._make_sets(seed)

    def resample_cell_drug_balanced(self, seed: int) -> None:
        """Uniform pair, uniform dose, then cell-count-weighted context.

        Only retained training rows are eligible. Each draw stays homogeneous
        in dose, batch, perturbation and time; epoch exposure stays unchanged.
        """
        rng = np.random.default_rng(int(seed))
        hierarchy = {}
        for key in sorted(self._regroup_pool):
            pair = (key[0], key[2])
            dose = (key[4], key[6])
            hierarchy.setdefault(pair, {}).setdefault(dose, []).append(key)
        pairs = sorted(hierarchy)
        prepared = {}
        for pair, doses in hierarchy.items():
            prepared[pair] = []
            for dose in sorted(doses):
                keys = doses[dose]
                counts = np.asarray([len(self._regroup_pool[k]) for k in keys])
                prepared[pair].append((keys, counts / counts.sum()))
        sets = []
        pair_counts = dict.fromkeys(pairs, 0)
        for _ in range(len(self.sets)):
            pair = pairs[int(rng.integers(len(pairs)))]
            doses = prepared[pair]
            keys, probabilities = doses[int(rng.integers(len(doses)))]
            key = keys[int(rng.choice(len(keys), p=probabilities))]
            pool = self._regroup_pool[key]
            if len(pool) >= self.set_size:
                indices = rng.choice(pool, size=self.set_size, replace=False)
            else:
                indices = np.concatenate([
                    rng.permutation(pool),
                    rng.choice(pool, size=self.set_size - len(pool), replace=True),
                ])
            sets.append(indices)
            pair_counts[pair] += 1
        self.sets = sets
        print(
            f"Cell-drug balanced sampling: seed={seed}, pairs={len(pairs)}, "
            f"sets={len(sets)}, pair_draws_min={min(pair_counts.values())}, "
            f"pair_draws_max={max(pair_counts.values())}", flush=True,
        )

    def __len__(self) -> int:
        return len(self.sets)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_indices = self.sets[item]
        first = int(target_indices[0])
        context = (
            int(self.store.cell_idx[first]),
            int(self.store.batch_idx[first]),
        )
        pool = self.store.control_pools[context]
        selected = torch.randint(len(pool), (self.set_size,)).numpy()
        control_indices = pool[selected]

        def repeated(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(values[target_indices], dtype=dtype)

        # A short condition pool is padded by resampling cells already in the
        # chunk, so every duplicate index marks a padded slot. Flagging the
        # first occurrence of each index reconstructs the padding exactly: the
        # real part of a chunk is a permutation of distinct cells. PerturbDiff
        # carries the same flag and drops padded cells from its loss; without
        # it a 16-slot set built from 1 real cell has zero population variance
        # and still trains the distribution term.
        _, first_occurrence = np.unique(target_indices, return_index=True)
        valid = np.zeros(len(target_indices), dtype=bool)
        valid[first_occurrence] = True

        return {
            "expression": torch.from_numpy(
                self.store.expression.rows(target_indices)
            ),
            "valid": torch.from_numpy(valid),
            "source": torch.from_numpy(
                self.store.expression.rows(control_indices)
            ),
            # Stable response origin from every matching training control;
            # sampled controls remain stochastic population-attention context.
            "control_mean": torch.from_numpy(
                self.store.control_mean[context]
            ).unsqueeze(0).expand(self.set_size, -1).clone(),
            "cell_idx": repeated(self.store.cell_idx, torch.long),
            "batch_idx": repeated(self.store.batch_idx, torch.long),
            "local_drug_idx": repeated(
                self.store.local_drug_idx, torch.long
            ),
            "pert_idx": repeated(self.store.pert_idx, torch.long),
            "dosage": repeated(self.store.dosage, torch.float32),
            "treatment_time": repeated(
                self.store.treatment_time, torch.float32
            ),
            "dose_idx": repeated(self.store.dose_idx, torch.long),
            "time_idx": repeated(self.store.time_idx, torch.long),
        }
