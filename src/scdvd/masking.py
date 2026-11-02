from __future__ import annotations

import numpy as np


class EmpiricalSupportLibrary:
    """Nearest-condition empirical joint masks used by Doloris sampling."""

    def __init__(
        self,
        store,
        *,
        split: str = "train",
        chunk_size: int = 4096,
        grouping: str = "scdvd",
    ) -> None:
        if grouping not in {"scdvd", "doloris"}:
            raise ValueError("grouping must be 'scdvd' or 'doloris'")
        mask = (store.split == split) & (store.control == 0)
        groups: dict[tuple, list[int]] = {}
        for index in np.where(mask)[0]:
            if grouping == "scdvd":
                key = (
                    int(store.cell_idx[index]),
                    int(store.batch_idx[index]),
                    int(store.local_drug_idx[index]),
                    float(store.dosage[index]),
                    float(store.treatment_time[index]),
                    int(store.pert_idx[index]),
                )
            else:
                # Match DoloRiS cond_dict construction: pool every training
                # cell sharing cell type, perturbation and dose, irrespective
                # of batch/replicate. local_drug_idx identifies the molecule.
                key = (
                    int(store.cell_idx[index]),
                    int(store.local_drug_idx[index]),
                    float(store.dosage[index]),
                )
            groups.setdefault(key, []).append(int(index))
        self.grouping = grouping
        self.store = store
        self.keys = tuple(sorted(groups))
        key_to_index = {key: index for index, key in enumerate(self.keys)}
        self.indices = {
            key: np.asarray(groups[key], dtype=np.int64) for key in self.keys
        }
        group_ids = np.full(len(store.split), -1, dtype=np.int64)
        for key, indices in self.indices.items():
            group_ids[indices] = key_to_index[key]
        counts = np.zeros(
            (len(self.keys), store.metadata.num_genes), dtype=np.uint32
        )
        totals = np.zeros(len(self.keys), dtype=np.int64)
        # Stream the CSR matrix once in contiguous blocks instead of issuing
        # hundreds of thousands of random HDF5 row reads.
        for start in range(0, len(store.split), int(chunk_size)):
            stop = min(start + int(chunk_size), len(store.split))
            local_group_ids = group_ids[start:stop]
            selected = local_group_ids >= 0
            if not selected.any():
                continue
            support = store.expression.row_block(start, stop)[selected] > 0
            selected_ids = local_group_ids[selected]
            for group_id in np.unique(selected_ids):
                local = selected_ids == group_id
                counts[group_id] += support[local].sum(axis=0, dtype=np.uint32)
                totals[group_id] += int(local.sum())
        self.probabilities = counts.astype(np.float32) / totals[:, None]

    def sample(
        self,
        predicted_probability: np.ndarray,
        *,
        count: int,
        rng: np.random.Generator,
        topk_conditions: int = 1,
        feature_topk: int = 300,
        high: float = 0.99,
        low: float = 0.01,
    ) -> np.ndarray:
        predicted = np.asarray(predicted_probability, dtype=np.float32).reshape(-1)
        feature_count = min(int(feature_topk), len(predicted))
        features = np.argsort(predicted)[-feature_count:]
        distances = np.linalg.norm(
            self.probabilities[:, features] - predicted[features][None, :],
            axis=1,
        )
        nearest = np.argsort(distances)[: int(topk_conditions)]
        pool = np.concatenate(
            [self.indices[self.keys[index]] for index in nearest]
        )
        selected = rng.choice(pool, size=int(count), replace=True)
        sampled = (self.store.expression.rows(selected) > 0).astype(np.float32)
        sampled[:, predicted >= float(high)] = 1.0
        sampled[:, predicted <= float(low)] = 0.0
        return sampled
