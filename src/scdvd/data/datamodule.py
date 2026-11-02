from __future__ import annotations

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from .unpaired import UnpairedH5ADStore


class DiffusionDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        dataset,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = True,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
        mixed_rows: bool = False,
        validation_split: str = "valid",
        validation_fraction: float = 0.0,
        validation_seed: int = 1337,
        population_set_size: int | None = None,
        regroup_each_epoch: bool = False,
        sampling_strategy: str = "sets",
        partition_set_size: int | None = None,
        drop_last_train: bool = True,
    ) -> None:
        super().__init__()
        self.dataset_config = dict(dataset)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.mixed_rows = bool(mixed_rows)
        self.validation_split = str(validation_split)
        self.validation_fraction = float(validation_fraction)
        self.validation_seed = int(validation_seed)
        self.population_set_size = (
            None if population_set_size is None else int(population_set_size)
        )
        self.regroup_each_epoch = bool(regroup_each_epoch)
        self.drop_last_train = bool(drop_last_train)
        self.sampling_strategy = str(sampling_strategy)
        self.partition_set_size = (
            None if partition_set_size is None else int(partition_set_size)
        )
        if self.partition_set_size is not None and (
            self.partition_set_size < 2
            or self.validation_fraction <= 0
        ):
            raise ValueError("partition_set_size requires held-out validation")
        if self.sampling_strategy not in {"sets", "cell_drug_balanced"}:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        if self.sampling_strategy != "sets" and self.population_set_size is None:
            raise ValueError("Balanced sampling requires population sets")
        self._train_loader_epoch = 0
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.store: UnpairedH5ADStore | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.store is None:
            self.store = UnpairedH5ADStore(**self.dataset_config)
        if stage in {None, "fit"}:
            if self.population_set_size is not None:
                dataset_factory = lambda split, max_samples: (
                    self.store.population_dataset(
                        split,
                        set_size=self.partition_set_size or self.population_set_size,
                        max_samples=max_samples,
                        seed=self.validation_seed,
                    )
                )
            else:
                dataset_factory = (
                    self.store.mixed_dataset
                    if self.mixed_rows
                    else self.store.dataset
                )
            if self.validation_fraction > 0:
                self.train_dataset = dataset_factory("train", max_samples=None)
                self.valid_dataset = dataset_factory("train", max_samples=None)

                if self.population_set_size is not None:
                    rng = np.random.default_rng(self.validation_seed)
                    order = rng.permutation(len(self.train_dataset.sets))
                    num_valid = max(
                        1,
                        int(round(len(order) * self.validation_fraction)),
                    )
                    valid_positions = order[:num_valid]
                    train_positions = order[num_valid:]
                    if not len(train_positions):
                        raise ValueError(
                            "validation_fraction leaves no population sets for training"
                        )
                    train_sets = [
                        self.train_dataset.sets[int(i)] for i in train_positions
                    ]
                    valid_sets = [
                        self.valid_dataset.sets[int(i)] for i in valid_positions
                    ]
                    if self.max_train_samples is not None:
                        maximum = max(
                            1,
                            int(self.max_train_samples)
                            // (self.partition_set_size or self.population_set_size),
                        )
                        train_sets = train_sets[:maximum]
                    if self.max_eval_samples is not None:
                        maximum = max(
                            1,
                            int(self.max_eval_samples)
                            // (self.partition_set_size or self.population_set_size),
                        )
                        valid_sets = valid_sets[:maximum]
                    self.train_dataset.sets = train_sets
                    self.valid_dataset.sets = valid_sets
                    self.train_dataset.restrict_regroup_pool_to_sets(train_sets)
                    # Keep the reference partition, validation sets and epoch
                    # draw count fixed when comparing training population sizes.
                    # The balanced sampler replaces these train sets before use.
                    self.train_dataset.set_size = self.population_set_size
                    if self.partition_set_size is not None and self.sampling_strategy == "sets":
                        # Fix reference cell membership, then group within each partition.
                        self.valid_dataset.restrict_regroup_pool_to_sets(valid_sets)
                        self.valid_dataset.set_size = self.population_set_size
                        self.train_dataset.regroup(self.validation_seed)
                        self.valid_dataset.regroup(self.validation_seed)
                else:
                    all_indices = self.train_dataset.indices.copy()
                    # Keep every control in training so control anchors/statistics
                    # remain well supported. Hold out only treated rows for model
                    # selection, without touching the external test split.
                    treated = all_indices[self.store.control[all_indices] == 0]
                    controls = all_indices[self.store.control[all_indices] == 1]
                    rng = np.random.default_rng(self.validation_seed)
                    treated = rng.permutation(treated)
                    num_valid = max(
                        1,
                        int(round(len(treated) * self.validation_fraction)),
                    )
                    valid_indices = treated[:num_valid]
                    train_indices = np.concatenate(
                        [controls, treated[num_valid:]]
                    )
                    train_indices = rng.permutation(train_indices)

                    if self.max_train_samples is not None:
                        train_indices = train_indices[
                            : int(self.max_train_samples)
                        ]
                    if self.max_eval_samples is not None:
                        valid_indices = valid_indices[
                            : int(self.max_eval_samples)
                        ]
                    self.train_dataset.indices = train_indices
                    self.valid_dataset.indices = valid_indices
            else:
                self.train_dataset = dataset_factory(
                    "train", max_samples=self.max_train_samples
                )
                self.valid_dataset = dataset_factory(
                    self.validation_split, max_samples=self.max_eval_samples
                )
        if stage in {None, "test"}:
            self.test_dataset = (
                self.store.population_dataset(
                    "test",
                    set_size=self.population_set_size,
                    max_samples=self.max_eval_samples,
                    seed=self.validation_seed,
                )
                if self.population_set_size is not None
                else self.store.dataset(
                    "test", max_samples=self.max_eval_samples
                )
            )

    @property
    def metadata(self):
        if self.store is None:
            raise RuntimeError("Call setup() before accessing dataset metadata")
        return self.store.metadata

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=shuffle and self.drop_last_train,
        )

    def train_dataloader(self):
        if self.sampling_strategy == "cell_drug_balanced":
            self.train_dataset.resample_cell_drug_balanced(
                self.validation_seed + self._train_loader_epoch
            )
            self._train_loader_epoch += 1
        elif self.regroup_each_epoch:
            regroup_seed = self.validation_seed + self._train_loader_epoch
            self.train_dataset.regroup(regroup_seed)
            print(
                "Regrouped training population sets: "
                f"epoch={self._train_loader_epoch}, seed={regroup_seed}, "
                f"sets={len(self.train_dataset.sets)}",
                flush=True,
            )
            self._train_loader_epoch += 1
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
