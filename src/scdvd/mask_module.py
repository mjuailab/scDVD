from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from .models.condition import DoseGatedMolecularEncoder
from .models.mask_compat import SparsityMaskPredictor


class EmpiricalMaskModule(L.LightningModule):
    """Separately optimized support predictor, as in Doloris."""

    def __init__(self, *, model_config: dict, training_config: dict, metadata: dict):
        super().__init__()
        self.save_hyperparameters()
        self.training_config = training_config
        chemical = torch.from_numpy(
            np.load(model_config["chemical_feature_path"], allow_pickle=False)
        ).float()
        condition_config = dict(model_config["condition"])
        condition_output_dim = int(condition_config["output_dim"])
        self.condition_encoder = DoseGatedMolecularEncoder(
            chemical_features=chemical,
            num_local_drugs=metadata["num_local_drugs"],
            output_dim=condition_output_dim,
            dropout=float(condition_config.get("dropout", 0.1)),
        )
        self.mask_predictor = SparsityMaskPredictor(
            num_genes=metadata["num_genes"],
            num_cell_types=metadata["num_cell_types"],
            num_batches=metadata["num_batches"],
            perturbation_dim=condition_output_dim,
            **dict(model_config["mask"]),
        )

    def perturbation_context(self, batch):
        return self.condition_encoder(
            pert_idx=batch["pert_idx"],
            local_drug_idx=batch["local_drug_idx"],
            dosage=batch["dosage"],
        )

    def _shared_step(self, batch, stage: str):
        context = self.perturbation_context(batch)
        logits = self.mask_predictor(
            batch["source"].float(),
            cell_idx=batch["cell_idx"],
            batch_idx=batch["batch_idx"],
            perturbation=context,
            control_mean=batch["control_mean"],
            control_std=batch["control_std"],
        )
        labels = (batch["target"] > 0).float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log(
            f"{stage}/mask_bce",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    @torch.no_grad()
    def predict_probabilities(self, batch) -> torch.Tensor:
        context = self.perturbation_context(batch)
        logits = self.mask_predictor(
            batch["source"].float(),
            cell_idx=batch["cell_idx"],
            batch_idx=batch["batch_idx"],
            perturbation=context,
            control_mean=batch["control_mean"],
            control_std=batch["control_std"],
        )
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.training_config.get("mask_learning_rate", 1e-2)),
            weight_decay=float(
                self.training_config.get("mask_weight_decay", 1e-2)
            ),
        )
