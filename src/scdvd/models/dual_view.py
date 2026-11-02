from __future__ import annotations

import torch
from torch import nn

from .common import timestep_embedding


def _modulate(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale) + shift


class CellViewBlock(nn.Module):
    """Conditioned per-cell residual MLP without population mixing."""

    def __init__(self, hidden_dim: int, *, mlp_ratio: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        width = hidden_dim * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3)
        )

    def forward(
        self, values: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return values + gate * self.mlp(
            _modulate(self.norm(values), shift, scale)
        )


class PopulationViewBlock(nn.Module):
    """Joint attention over two unpaired, exchangeable cell populations."""

    def __init__(
        self, hidden_dim: int, num_heads: int, *, mlp_ratio: int
    ) -> None:
        super().__init__()
        self.target_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.control_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.target_mlp_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )
        self.control_mlp_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )
        width = hidden_dim * int(mlp_ratio)
        self.target_mlp = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )
        self.control_mlp = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, hidden_dim),
        )
        self.target_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.control_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )

    def forward(
        self,
        target: torch.Tensor,
        control: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ts, tc, tg, tms, tmc, tmg = self.target_modulation(
            condition
        ).chunk(6, dim=-1)
        cs, cc, cg, cms, cmc, cmg = self.control_modulation(
            condition
        ).chunk(6, dim=-1)

        target_attention = _modulate(
            self.target_norm(target), ts, tc
        )
        control_attention = _modulate(
            self.control_norm(control), cs, cc
        )
        set_size = target.shape[1]
        joint = torch.cat([target_attention, control_attention], dim=1)
        joint, _ = self.attention(joint, joint, joint, need_weights=False)
        target_update = joint[:, :set_size]
        control_update = joint[:, set_size:]
        target = target + tg * target_update
        control = control + cg * control_update
        target = target + tmg * self.target_mlp(
            _modulate(self.target_mlp_norm(target), tms, tmc)
        )
        control = control + cmg * self.control_mlp(
            _modulate(self.control_mlp_norm(control), cms, cmc)
        )
        return target, control


class DualViewDenoiser(nn.Module):
    """Cell- and population-view denoiser for control-anchored responses."""

    def __init__(
        self,
        *,
        num_genes: int,
        num_cell_types: int,
        num_batches: int,
        perturbation_dim: int,
        hidden_dim: int = 256,
        cell_layers: int = 4,
        population_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        condition_dropout: float = 0.2,
        control_dropout: float = 0.0,
        population_gate_min: float = 0.05,
        population_gate_max: float = 0.50,
        population_gate_initial: float = 0.10,
        population_correction_rms_ratio: float | None = 0.50,
        population_correction_reference: str = "cell_deviation",
        population_correction_eps: float = 1e-6,
        detach_population_context: bool = True,
        population_gate_enabled: bool = True,
        center_cell_deviation: bool = True,
        center_population_correction: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_genes = int(num_genes)
        self.hidden_dim = int(hidden_dim)
        self.condition_dropout = float(condition_dropout)
        self.control_dropout = float(control_dropout)
        # Historical switches remain loadable; paper configs explicitly disable
        # the population gate and RMS cap, while retaining both centerings.
        self.population_gate_enabled = bool(population_gate_enabled)
        self.center_cell_deviation = bool(center_cell_deviation)
        self.center_population_correction = bool(center_population_correction)
        if self.population_gate_enabled:
            if not 0.0 <= population_gate_min < population_gate_max <= 1.0:
                raise ValueError(
                    "population gate bounds must satisfy 0 <= min < max <= 1"
                )
            if not (
                population_gate_min
                < population_gate_initial
                < population_gate_max
            ):
                raise ValueError(
                    "population_gate_initial must lie inside its bounds"
                )
        if (
            population_correction_rms_ratio is not None
            and population_correction_rms_ratio <= 0
        ):
            raise ValueError(
                "population_correction_rms_ratio must be positive or null"
            )
        if population_correction_eps <= 0:
            raise ValueError("population_correction_eps must be positive")
        self.population_gate_min = float(population_gate_min)
        self.population_gate_max = float(population_gate_max)
        self.population_correction_rms_ratio = (
            None
            if population_correction_rms_ratio is None
            else float(population_correction_rms_ratio)
        )
        # The RMS budget for the population correction was historically a
        # fraction of the cell deviation. That ties the model's capacity for
        # spread to a quantity the reconstruction MSE is actively driving to
        # zero: the better the mean fit, the less room the correction has.
        # "control_spread" measures the budget against how far the real control
        # cells sit from their own mean instead, which the MSE does not shrink.
        if population_correction_reference not in {
            "cell_deviation",
            "control_spread",
        }:
            raise ValueError(
                "population_correction_reference must be 'cell_deviation' "
                "or 'control_spread'"
            )
        self.population_correction_reference = str(
            population_correction_reference
        )
        self.population_correction_eps = float(population_correction_eps)
        self.detach_population_context = bool(detach_population_context)

        self.target_projection = nn.Linear(num_genes * 2, hidden_dim)
        self.control_projection = nn.Linear(num_genes, hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.perturbation_projection = nn.Linear(
            perturbation_dim, hidden_dim
        )
        self.cell_embedding = nn.Embedding(num_cell_types, hidden_dim)
        self.batch_embedding = nn.Embedding(num_batches, hidden_dim)
        self.null_perturbation = nn.Parameter(
            torch.zeros(1, 1, hidden_dim)
        )
        # Explicitly mark target-side noisy responses and source controls
        # after their separate projections and before joint set attention.
        self.population_type_embedding = nn.Embedding(2, hidden_dim)

        self.cell_blocks = nn.ModuleList(
            [
                CellViewBlock(hidden_dim, mlp_ratio=mlp_ratio)
                for _ in range(int(cell_layers))
            ]
        )
        self.population_blocks = nn.ModuleList(
            [
                PopulationViewBlock(
                    hidden_dim, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(int(population_layers))
            ]
        )
        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.population_norm = nn.LayerNorm(hidden_dim)
        self.cell_output = nn.Linear(hidden_dim, num_genes)
        self.population_correction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_genes),
        )
        self.population_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mean_response = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_genes),
        )
        # Start as a cell-level model. The population correction is learned
        # only when the distributional signal supports it.
        nn.init.zeros_(self.population_correction[-1].weight)
        nn.init.zeros_(self.population_correction[-1].bias)
        nn.init.zeros_(self.population_gate[-1].weight)
        if self.population_gate_enabled:
            gate_fraction = (
                (float(population_gate_initial) - self.population_gate_min)
                / (self.population_gate_max - self.population_gate_min)
            )
            initial_gate_logit = torch.logit(torch.tensor(gate_fraction))
            nn.init.constant_(
                self.population_gate[-1].bias, initial_gate_logit.item()
            )
        else:
            nn.init.zeros_(self.population_gate[-1].bias)
        nn.init.zeros_(self.mean_response[-1].weight)
        nn.init.zeros_(self.mean_response[-1].bias)
        # Preserve shared initialization/RNG, then remove unused gate parameters.
        if not self.population_gate_enabled:
            del self.population_gate

    def forward(
        self,
        noisy_response: torch.Tensor,
        centered_control: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        perturbation: torch.Tensor,
        cell_idx: torch.Tensor,
        batch_idx: torch.Tensor,
        self_condition: torch.Tensor | None = None,
        force_unconditional: bool = False,
        force_conditional: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self_condition is None:
            self_condition = torch.zeros_like(noisy_response)
        target_base = self.target_projection(
            torch.cat([noisy_response, self_condition], dim=-1)
        )

        control_values = centered_control
        if self.training and self.control_dropout > 0:
            dropped = (
                torch.rand(
                    centered_control.shape[0],
                    1,
                    1,
                    device=centered_control.device,
                )
                < self.control_dropout
            )
            control_values = torch.where(
                dropped, torch.zeros_like(control_values), control_values
            )
        control_hidden = self.control_projection(control_values)
        control_hidden = control_hidden + self.population_type_embedding.weight[
            0
        ].view(1, 1, -1)

        perturbation_hidden = self.perturbation_projection(perturbation)
        if force_unconditional:
            perturbation_hidden = self.null_perturbation.expand_as(
                perturbation_hidden
            )
        elif (
            self.training
            and not force_conditional
            and self.condition_dropout > 0
        ):
            dropped = (
                torch.rand(
                    perturbation_hidden.shape[0],
                    1,
                    1,
                    device=perturbation_hidden.device,
                )
                < self.condition_dropout
            )
            perturbation_hidden = torch.where(
                dropped,
                self.null_perturbation.expand_as(perturbation_hidden),
                perturbation_hidden,
            )

        time = self.time_projection(
            timestep_embedding(timesteps, self.hidden_dim)
        ).unsqueeze(1)
        mean_condition = (
            perturbation_hidden
            + self.cell_embedding(cell_idx.long())
            + self.batch_embedding(batch_idx.long())
        )
        condition = mean_condition + time

        cell_hidden = target_base
        for block in self.cell_blocks:
            cell_hidden = block(cell_hidden, condition)

        # Treat the cell encoding and condition as read-only context for the
        # population view. This keeps distribution-only gradients out of the
        # already-strong cell denoising path while retaining the same features.
        population_target_base = (
            target_base.detach()
            if self.detach_population_context
            else target_base
        )
        population_hidden = population_target_base + self.population_type_embedding.weight[
            1
        ].view(1, 1, -1)
        population_condition = (
            condition.detach()
            if self.detach_population_context
            else condition
        )
        for block in self.population_blocks:
            population_hidden, control_hidden = block(
                population_hidden, control_hidden, population_condition
            )

        raw_cell_deviation = self.cell_output(self.cell_norm(cell_hidden))
        cell_deviation = raw_cell_deviation
        if self.center_cell_deviation:
            cell_deviation = raw_cell_deviation - raw_cell_deviation.mean(
                dim=1, keepdim=True
            )
        # The population mean cannot be copied from noisy treated cells: it is
        # predicted directly from drug/KG/dose/treatment-time and cell/batch
        # context. Diffusion timestep is intentionally excluded here.
        mean_response = self.mean_response(
            mean_condition.mean(dim=1)
        ).unsqueeze(1)
        cell_response = mean_response + cell_deviation
        normalized_population = self.population_norm(population_hidden)
        cell_correction_context = self.cell_norm(cell_hidden)
        if self.detach_population_context:
            cell_correction_context = cell_correction_context.detach()
        raw_correction = self.population_correction(
            torch.cat(
                [
                    cell_correction_context,
                    normalized_population,
                ],
                dim=-1,
            )
        )
        # The population view models heterogeneity rather than silently
        # replacing the condition-level mean response. Center every gene over
        # the exchangeable cell set before bounding its residual RMS.
        centered_correction = raw_correction
        if self.center_population_correction:
            centered_correction = raw_correction - raw_correction.mean(
                dim=1, keepdim=True
            )
        # Put epsilon inside sqrt. At the zero-initialized correction head,
        # sqrt(mean(x^2)) has an infinite derivative at x=0 and corrupts the
        # first optimizer update even though its forward value is finite.
        rms_epsilon_squared = self.population_correction_eps**2
        correction_rms = (
            centered_correction.square().mean(
                dim=(1, 2), keepdim=True
            )
            + rms_epsilon_squared
        ).sqrt()
        correction_reference = (
            centered_control
            if self.population_correction_reference == "control_spread"
            else cell_deviation
        )
        cell_reference_rms = (
            correction_reference.detach().square().mean(
                dim=(1, 2), keepdim=True
            )
            + rms_epsilon_squared
        ).sqrt()
        if self.population_correction_rms_ratio is None:
            correction_rms_limit = torch.full_like(
                cell_reference_rms, float("inf")
            )
            correction_scale = torch.ones_like(correction_rms)
            bounded_correction = centered_correction
        else:
            correction_rms_limit = (
                self.population_correction_rms_ratio * cell_reference_rms
            ).clamp_min(self.population_correction_eps)
            correction_scale = (
                correction_rms_limit
                / (correction_rms + self.population_correction_eps)
            ).clamp(max=1.0)
            bounded_correction = centered_correction * correction_scale

        # A free sigmoid gate previously collapsed to zero, while a fixed 1.0
        # gate exposed unbounded corrections. Keep it learnable but constrain
        # it to a non-zero, sub-unit interval; the RMS cap prevents the
        # correction head from compensating by growing without bound.
        gate_context = torch.cat(
            [
                normalized_population.mean(dim=1),
                population_condition.mean(dim=1),
            ],
            dim=-1,
        )
        if self.population_gate_enabled:
            gate_fraction = torch.sigmoid(self.population_gate(gate_context))
            population_gate = (
                self.population_gate_min
                + (self.population_gate_max - self.population_gate_min)
                * gate_fraction
            ).unsqueeze(1)
        else:
            population_gate = torch.ones(
                gate_context.shape[0],
                1,
                1,
                device=gate_context.device,
                dtype=bounded_correction.dtype,
            )
        correction = population_gate * bounded_correction
        response = cell_response + correction
        if return_aux:
            return {
                "response": response,
                "cell_response": cell_response,
                "cell_deviation": cell_deviation,
                "mean_response": mean_response,
                "population_correction": correction,
                "raw_population_correction": raw_correction,
                "population_gate": population_gate,
                "population_correction_scale": correction_scale,
                "population_correction_rms_limit": (
                    population_gate * correction_rms_limit
                ),
            }
        return response
