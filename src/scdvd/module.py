from __future__ import annotations

import lightning as L
import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .models.condition import DoseGatedMolecularEncoder
from .models.dual_view import DualViewDenoiser


# Re-export these functions for existing analysis imports.
from .losses import (
    positive_expression_mse, support_weighted_expression_mse,
    geomloss_energy_distance, raw_energy_distance, multiscale_rbf_mmd,
)


class scDVDModule(L.LightningModule):
    """Control-anchored dual-view response diffusion with a two-term loss."""

    def __init__(
        self, *, model_config: dict, training_config: dict, metadata: dict
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.training_config = training_config
        self.mode = "scdvd"
        self.population_set_size = int(model_config["population"]["set_size"])

        chemical = torch.from_numpy(
            np.load(model_config["chemical_feature_path"], allow_pickle=False)
        ).float()
        condition = dict(model_config["condition"])
        condition_dim = int(condition["output_dim"])
        self.condition_encoder = DoseGatedMolecularEncoder(
            chemical_features=chemical,
            num_local_drugs=metadata["num_local_drugs"],
            output_dim=condition_dim,
            dropout=float(condition.get("dropout", 0.1)),
        )
        self.denoiser = DualViewDenoiser(
            num_genes=metadata["num_genes"],
            num_cell_types=metadata["num_cell_types"],
            num_batches=metadata["num_batches"],
            perturbation_dim=condition_dim,
            **dict(model_config["denoiser"]),
        )
        self.diffusion = GaussianDiffusion(**dict(model_config["diffusion"]))

        objective = dict(model_config["objective"])
        self.distribution_loss = str(
            objective.get("distribution_loss", "energy")
        ).lower()
        if self.distribution_loss not in {"energy", "mmd", "geomloss_energy", "none"}:
            raise ValueError(
                "objective.distribution_loss must be 'energy', 'mmd', 'geomloss_energy', or 'none'"
            )
        self.energy_weight = float(objective.get("energy_weight", 0.005))
        self.energy_transform = str(objective.get("energy_transform", "log1p"))
        if self.energy_transform not in {"log1p", "identity"}:
            raise ValueError("objective.energy_transform must be 'log1p' or 'identity'")
        self.mmd_weight = float(objective.get("mmd_weight", 0.1))
        self.mmd_bandwidth_scales = tuple(
            float(value)
            for value in objective.get(
                "mmd_bandwidth_scales", [0.5, 1.0, 2.0, 4.0]
            )
        )
        self.self_condition_probability = float(
            objective.get("self_condition_probability", 0.25)
        )
        self.guidance_scale = float(objective.get("guidance_scale", 1.5))
        self.zero_expression_weight = float(
            objective.get("zero_expression_weight", 0.0)
        )
        if not 0.0 <= self.zero_expression_weight <= 1.0:
            raise ValueError("objective.zero_expression_weight must be in [0, 1]")
        self.distribution_support = str(
            objective.get("distribution_support", "raw")
        ).lower()
        if self.distribution_support not in {"raw", "target"}:
            raise ValueError(
                "objective.distribution_support must be 'raw' or 'target'"
            )
        # Padding has no direct loss weight; it still participates in attention.
        self.mask_padded_cells = bool(
            objective.get("mask_padded_cells", False)
        )
        # The paper evaluates the energy loss at the sampled training timestep.
        self.distribution_timestep = str(
            objective.get("distribution_timestep", "sampled")
        ).lower()
        if self.distribution_timestep not in {"sampled", "terminal"}:
            raise ValueError(
                "objective.distribution_timestep must be 'sampled' or 'terminal'"
            )
        self.response_scale = float(objective.get("response_scale", 1.0))
        if self.response_scale <= 0:
            raise ValueError("objective.response_scale must be positive")
        self.fail_on_nonfinite = bool(
            training_config.get("fail_on_nonfinite", True)
        )

    @staticmethod
    def control_anchor(
        source: torch.Tensor, control_mean: torch.Tensor
    ) -> torch.Tensor:
        if source.shape != control_mean.shape:
            raise ValueError("source and control_mean must share [B,S,G] shape")
        return control_mean.float()

    def perturbation_context(self, batch: dict) -> torch.Tensor:
        shape = batch["pert_idx"].shape
        encoded = self.condition_encoder(
            pert_idx=batch["pert_idx"][:, 0],
            local_drug_idx=batch["local_drug_idx"][:, 0],
            dosage=batch["dosage"][:, 0],
        )
        return encoded.unsqueeze(1).expand(-1, shape[1], -1)

    def _predict_response(
        self,
        batch: dict,
        noisy_response: torch.Tensor,
        model_t: torch.Tensor,
        perturbation: torch.Tensor,
        centered_control: torch.Tensor,
    ) -> torch.Tensor:
        self_condition = None
        if (
            self.training
            and self.self_condition_probability > 0
            and torch.rand((), device=noisy_response.device)
            < self.self_condition_probability
        ):
            with torch.no_grad():
                self_condition = self.denoiser(
                    noisy_response,
                    centered_control,
                    model_t,
                    perturbation=perturbation,
                    cell_idx=batch["cell_idx"],
                    batch_idx=batch["batch_idx"],
                )
        return self.denoiser(
            noisy_response,
            centered_control,
            model_t,
            perturbation=perturbation,
            cell_idx=batch["cell_idx"],
            batch_idx=batch["batch_idx"],
            self_condition=self_condition,
        )

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        target = batch["expression"].float()
        source = batch["source"].float()
        anchor = self.control_anchor(source, batch["control_mean"])
        clean_response = (target - anchor) * self.response_scale
        centered_control = (source - anchor) * self.response_scale

        diffusion_t = self.diffusion.random_timesteps(
            target.shape[0], target.device
        )
        noisy_response = self.diffusion.q_sample(clean_response, diffusion_t)
        prediction = anchor + self._predict_response(
            batch,
            noisy_response,
            self.diffusion.model_timesteps(diffusion_t),
            self.perturbation_context(batch),
            centered_control,
        ) / self.response_scale

        valid = (
            batch["valid"]
            if self.mask_padded_cells and "valid" in batch
            else None
        )
        expression_loss = support_weighted_expression_mse(
            prediction,
            target,
            zero_weight=self.zero_expression_weight,
            valid=valid,
        ).mean()
        if self.distribution_timestep == "terminal":
            terminal_t = torch.full_like(
                diffusion_t, self.diffusion.num_timesteps - 1
            )
            distribution_prediction = anchor + self._predict_response(
                batch,
                self.diffusion.q_sample(clean_response, terminal_t),
                self.diffusion.model_timesteps(terminal_t),
                self.perturbation_context(batch),
                centered_control,
            ) / self.response_scale
        else:
            distribution_prediction = prediction
        if self.distribution_support == "target":
            distribution_prediction = prediction * (target > 0).to(
                prediction.dtype
            )
        energy = torch.zeros((), device=target.device)
        energy_loss = torch.zeros((), device=target.device)
        mmd_squared = torch.zeros((), device=target.device)
        mmd_bandwidth = torch.zeros((), device=target.device)
        if self.distribution_loss == "energy":
            energy = raw_energy_distance(target, distribution_prediction, valid=valid)
            energy_loss = (
                torch.log1p(energy) if self.energy_transform == "log1p" else energy
            ).mean()
            distribution_loss = energy_loss
            weighted_distribution_loss = self.energy_weight * energy_loss
        elif self.distribution_loss == "geomloss_energy":
            mmd_squared = geomloss_energy_distance(
                target, distribution_prediction, valid=valid
            ).mean()
            distribution_loss = mmd_squared
            weighted_distribution_loss = self.mmd_weight * mmd_squared
        elif self.distribution_loss == "mmd":
            population_mmd, bandwidth = multiscale_rbf_mmd(
                target,
                distribution_prediction,
                bandwidth_scales=self.mmd_bandwidth_scales,
                valid=valid,
            )
            mmd_squared = population_mmd.mean()
            mmd_bandwidth = bandwidth.mean()
            distribution_loss = mmd_squared
            weighted_distribution_loss = self.mmd_weight * mmd_squared
        else:
            distribution_loss = torch.zeros((), device=target.device)
            weighted_distribution_loss = distribution_loss
        loss = expression_loss + weighted_distribution_loss

        if self.fail_on_nonfinite and not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss: "
                f"expression={expression_loss.item()}, "
                f"distribution={distribution_loss.item()}"
            )

        common = {
            "on_step": stage == "train",
            "on_epoch": True,
            "sync_dist": True,
            "batch_size": target.shape[0],
        }
        self.log(f"{stage}/loss", loss, prog_bar=True, **common)
        self.log(f"{stage}/expression_mse", expression_loss, **common)
        self.log(f"{stage}/raw_energy", energy.mean(), **common)
        self.log(f"{stage}/raw_energy_loss", energy_loss, **common)
        self.log(
            f"{stage}/weighted_raw_energy_loss",
            self.energy_weight * energy_loss,
            **common,
        )
        self.log(f"{stage}/mmd_squared", mmd_squared, **common)
        self.log(f"{stage}/mmd_bandwidth", mmd_bandwidth, **common)
        self.log(
            f"{stage}/weighted_mmd_loss",
            self.mmd_weight * mmd_squared,
            **common,
        )
        self.log(
            f"{stage}/weighted_distribution_loss",
            weighted_distribution_loss,
            **common,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    @torch.no_grad()
    def _sample_population(
        self,
        batch: dict,
        *,
        ddim_steps: int,
        clip_absolute_each_step: bool = False,
        latent_source: str = "noise",
    ) -> torch.Tensor:
        source = batch["source"].float().unsqueeze(0)
        set_batch = {
            key: value.unsqueeze(0)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and value.ndim == 1
        }
        anchor = self.control_anchor(
            source, batch["control_mean"].float().unsqueeze(0)
        )
        centered_control = (source - anchor) * self.response_scale
        perturbation = self.perturbation_context(set_batch)
        previous = None

        def model(noisy_response, timesteps):
            nonlocal previous
            conditional = self.denoiser(
                noisy_response,
                centered_control,
                timesteps,
                perturbation=perturbation,
                cell_idx=set_batch["cell_idx"],
                batch_idx=set_batch["batch_idx"],
                self_condition=previous,
                force_conditional=True,
            )
            unconditional = self.denoiser(
                noisy_response,
                centered_control,
                timesteps,
                perturbation=perturbation,
                cell_idx=set_batch["cell_idx"],
                batch_idx=set_batch["batch_idx"],
                self_condition=previous,
                force_unconditional=True,
            )
            guided = unconditional + self.guidance_scale * (
                conditional - unconditional
            )
            # DoloRiS clips its predicted clean expression at every reverse
            # step. scDVD predicts a control-anchored response, so apply the
            # equivalent constraint in absolute-expression space and then
            # transform it back to response space for the DDIM update.
            if clip_absolute_each_step:
                guided = (
                    (anchor + guided / self.response_scale).clamp(0.0, 1.0)
                    - anchor
                ) * self.response_scale
            previous = guided.detach()
            return guided

        if latent_source not in {"noise", "control_inversion"}:
            raise ValueError(
                "latent_source must be 'noise' or 'control_inversion'"
            )
        if latent_source == "control_inversion":
            # DoloRiS seeds every reverse trajectory with the DDIM inversion of
            # a real control cell rather than fresh Gaussian noise, so the
            # cell-to-cell heterogeneity of the control population is carried
            # into the generated population. Deterministic DDIM has no other
            # source of diversity. The response-space analogue of inverting a
            # control expression profile is inverting the centered control, and
            # the inversion runs under the null condition, matching DoloRiS
            # inverting at dosage zero.
            def unconditional_model(noisy_response, timesteps):
                return self.denoiser(
                    noisy_response,
                    centered_control,
                    timesteps,
                    perturbation=perturbation,
                    cell_idx=set_batch["cell_idx"],
                    batch_idx=set_batch["batch_idx"],
                    force_unconditional=True,
                )

            latent = self.diffusion.ddim_invert(
                unconditional_model,
                centered_control,
                steps=int(ddim_steps),
                clip=None,
            )
            previous = None
        else:
            latent = torch.randn_like(source)

        response = self.diffusion.ddim_sample(
            model, latent, steps=int(ddim_steps), clip=None
        )
        return (
            (anchor + response / self.response_scale).squeeze(0).clamp(0.0, 1.0)
        )

    @torch.no_grad()
    def bridge_sample(
        self,
        batch,
        *,
        ddim_steps=50,
        clip_absolute_each_step: bool = False,
        latent_source: str = "noise",
        **kwargs,
    ):
        del kwargs
        source = batch["source"]
        batch_idx = batch["batch_idx"]
        generated = torch.empty_like(source, dtype=torch.float32)
        for batch_value in torch.unique(batch_idx, sorted=True):
            positions = torch.nonzero(
                batch_idx == batch_value, as_tuple=False
            ).flatten()
            for start in range(0, len(positions), self.population_set_size):
                output_positions = positions[
                    start : start + self.population_set_size
                ]
                valid = int(output_positions.numel())
                chunk_positions = output_positions
                if valid < self.population_set_size:
                    needed = self.population_set_size - valid
                    repeats = (needed + len(positions) - 1) // len(positions)
                    padding = positions.repeat(repeats)[:needed]
                    chunk_positions = torch.cat([output_positions, padding])
                chunk = {
                    key: (
                        value.index_select(0, chunk_positions)
                        if isinstance(value, torch.Tensor)
                        and value.ndim > 0
                        and value.shape[0] == source.shape[0]
                        else value
                    )
                    for key, value in batch.items()
                }
                sampled = self._sample_population(
                    chunk,
                    ddim_steps=ddim_steps,
                    clip_absolute_each_step=bool(clip_absolute_each_step),
                    latent_source=str(latent_source),
                )
                generated.index_copy_(0, output_positions, sampled[:valid])
        return generated

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.training_config["learning_rate"]),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.training_config["max_epochs"]),
            eta_min=float(
                self.training_config.get("minimum_learning_rate", 0.0)
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
