from __future__ import annotations

import math

import torch
from torch import nn


class _DolorisMLP(nn.Module):
    """BatchNorm/SiLU/dropout MLP used by the original DoloRiS code."""

    def __init__(self, sizes: list[int], *, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(sizes) - 2:
                layers.extend(
                    [nn.BatchNorm1d(output_dim), nn.SiLU(), nn.Dropout(dropout)]
                )
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        singleton = values.shape[0] == 1 and self.training
        if singleton:
            values = values.repeat(2, 1)
        output = self.network(values)
        return output[:1] if singleton else output


class DoseGatedMolecularEncoder(nn.Module):
    """UniMol projection modulated by a learned scalar dose gate."""

    def __init__(
        self,
        *,
        chemical_features: torch.Tensor,
        num_local_drugs: int,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if chemical_features.ndim != 2:
            raise ValueError("UniMol chemical features must be rank-2")
        input_dim = int(chemical_features.shape[1])
        self.register_buffer(
            "chemical_features", chemical_features.float(), persistent=False
        )
        self.local_chemical = nn.Embedding(num_local_drugs, input_dim)
        self.molecule_mlp = _DolorisMLP(
            [input_dim, input_dim, int(output_dim)], dropout=dropout
        )
        self.dosager = _DolorisMLP(
            [int(output_dim) + 1, int(output_dim), int(output_dim), 1],
            dropout=dropout,
        )

    def forward(
        self,
        *,
        pert_idx: torch.Tensor,
        local_drug_idx: torch.Tensor,
        dosage: torch.Tensor,
    ) -> torch.Tensor:
        pert_idx = pert_idx.long().view(-1)
        local_drug_idx = local_drug_idx.long().view(-1)
        dose = dosage.float().view(-1, 1)
        valid = (pert_idx >= 0) & (pert_idx < self.chemical_features.shape[0])
        safe_global = pert_idx.clamp(0, self.chemical_features.shape[0] - 1)
        molecule = self.local_chemical(local_drug_idx)
        molecule = torch.where(
            valid.unsqueeze(1), self.chemical_features[safe_global], molecule
        )
        molecule = self.molecule_mlp(molecule)
        dosage_gate = torch.sigmoid(
            self.dosager(torch.cat([molecule, dose], dim=1))
        )
        embedding = dosage_gate * molecule
        return torch.where(dose != 0, embedding, torch.zeros_like(embedding))


class HetDrugKGConditionEncoder(nn.Module):
    """Chemical/KG/dose/time perturbation encoder for diffusion conditioning.

    Dose and treatment time are encoded as continuous covariates. Dataset-local
    learned features are used only when a global KG perturbation index is
    unavailable.
    """

    def __init__(
        self,
        *,
        chemical_features: torch.Tensor,
        biological_features: torch.Tensor,
        num_local_drugs: int,
        num_dose_bins: int | None = None,
        num_time_bins: int | None = None,
        token_dim: int = 512,
        covariate_hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_chemical: bool = True,
        use_biological: bool = True,
    ) -> None:
        super().__init__()
        if chemical_features.ndim != 2 or biological_features.ndim != 2:
            raise ValueError("Global chemical and biological features must be rank-2")
        if chemical_features.shape[0] != biological_features.shape[0]:
            raise ValueError("Chemical and biological features use different drug universes")
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")

        self.register_buffer(
            "chemical_features", chemical_features.float(), persistent=False
        )
        self.use_chemical = bool(use_chemical)
        self.use_biological = bool(use_biological)
        if not self.use_chemical and not self.use_biological:
            raise ValueError("At least one perturbation feature branch must be enabled")
        self.register_buffer(
            "biological_features", biological_features.float(), persistent=False
        )
        self.local_chemical = nn.Embedding(num_local_drugs, chemical_features.shape[1])
        self.local_biological = nn.Embedding(num_local_drugs, biological_features.shape[1])

        self.chemical_projection = nn.Linear(chemical_features.shape[1], token_dim)
        self.biological_projection = nn.Linear(biological_features.shape[1], token_dim)
        self.legacy_binned_covariates = num_dose_bins is not None
        if self.legacy_binned_covariates:
            if num_time_bins is None:
                raise ValueError("num_time_bins is required with num_dose_bins")
            self.dose_bin_embedding = nn.Embedding(num_dose_bins, token_dim)
            self.time_embedding = nn.Embedding(num_time_bins, token_dim)
            self.continuous_dose = nn.Sequential(
                nn.Linear(1, token_dim),
                nn.SiLU(),
                nn.Linear(token_dim, token_dim),
            )
        else:
            self.dose_encoder = nn.Sequential(
                nn.Linear(1, covariate_hidden_dim),
                nn.SiLU(),
                nn.Linear(covariate_hidden_dim, covariate_hidden_dim),
            )
            self.time_encoder = nn.Sequential(
                nn.Linear(1, covariate_hidden_dim),
                nn.SiLU(),
                nn.Linear(covariate_hidden_dim, covariate_hidden_dim),
            )
            self.covariate_fusion = nn.Sequential(
                nn.Linear(2 * covariate_hidden_dim, token_dim),
                nn.SiLU(),
                nn.Linear(token_dim, token_dim),
            )

        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))
        self.position_embedding = nn.Embedding(4, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.input_norm = nn.LayerNorm(token_dim)
        self.output = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, output_dim))
        nn.init.normal_(self.cls_token, std=1.0 / math.sqrt(token_dim))

    def forward(
        self,
        *,
        pert_idx: torch.Tensor,
        local_drug_idx: torch.Tensor,
        dosage: torch.Tensor,
        treatment_time: torch.Tensor | None = None,
        dose_idx: torch.Tensor | None = None,
        time_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pert_idx = pert_idx.long().view(-1)
        local_drug_idx = local_drug_idx.long().view(-1)
        valid = (pert_idx >= 0) & (pert_idx < self.chemical_features.shape[0])
        safe_global = pert_idx.clamp(0, self.chemical_features.shape[0] - 1)

        chemical = self.local_chemical(local_drug_idx)
        biological = self.local_biological(local_drug_idx)
        if valid.any():
            chemical = torch.where(
                valid.unsqueeze(1), self.chemical_features[safe_global], chemical
            )
            biological = torch.where(
                valid.unsqueeze(1), self.biological_features[safe_global], biological
            )

        chemical_token = self.chemical_projection(chemical)
        biological_token = self.biological_projection(biological)
        if not self.use_chemical:
            chemical_token = torch.zeros_like(chemical_token)
        if not self.use_biological:
            biological_token = torch.zeros_like(biological_token)
        if self.legacy_binned_covariates:
            if dose_idx is None or time_idx is None:
                raise ValueError("Legacy checkpoints require dose_idx and time_idx")
            dosage_value = torch.log1p(dosage.float().view(-1, 1).clamp_min(0.0))
            covariate_token = (
                self.continuous_dose(dosage_value)
                + self.dose_bin_embedding(
                    dose_idx.long().view(-1).clamp(
                        0, self.dose_bin_embedding.num_embeddings - 1
                    )
                )
                + self.time_embedding(
                    time_idx.long().view(-1).clamp(
                        0, self.time_embedding.num_embeddings - 1
                    )
                )
            )
        else:
            if treatment_time is None:
                raise ValueError("Continuous checkpoints require treatment_time")
            # Positive doses retain their absolute base-10 scale in micromolar.
            # Zero is reserved for controls or missing released doses and is
            # clamped solely to keep the transform finite.
            log_dose = torch.log10(dosage.float().view(-1, 1).clamp_min(1e-8))
            log_time = torch.log1p(treatment_time.float().view(-1, 1).clamp_min(0.0))
            dose_features = self.dose_encoder(log_dose)
            time_features = self.time_encoder(log_time)
            covariate_token = self.covariate_fusion(
                torch.cat([dose_features, time_features], dim=-1)
            )

        batch_size = pert_idx.shape[0]
        tokens = torch.stack(
            [chemical_token, biological_token, covariate_token], dim=1
        )
        tokens = torch.cat([self.cls_token.expand(batch_size, -1, -1), tokens], dim=1)
        positions = torch.arange(4, device=tokens.device).unsqueeze(0)
        tokens = self.input_norm(tokens + self.position_embedding(positions))
        encoded = self.transformer(tokens)
        return self.output(encoded[:, 0])
