from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape) -> torch.Tensor:
    output = values.gather(0, timesteps.long())
    return output.view(output.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Gaussian forward process plus deterministic DDIM sampling/inversion."""

    def __init__(
        self,
        *,
        num_timesteps: int = 500,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
        timestep_respacing: int | str | None = None,
    ) -> None:
        super().__init__()
        base_num_timesteps = int(num_timesteps)
        if schedule == "linear":
            start = float(beta_start)
            end = float(beta_end)
        elif schedule == "doloris_linear":
            scale = 1000.0 / base_num_timesteps
            start = scale * 1e-4
            end = scale * 2e-2
        else:
            raise ValueError("schedule must be 'linear' or 'doloris_linear'")
        base_betas = torch.linspace(
            start, end, base_num_timesteps, dtype=torch.float64
        )
        if timestep_respacing is None:
            timestep_map = torch.arange(base_num_timesteps, dtype=torch.long)
            betas = base_betas
        else:
            if isinstance(timestep_respacing, str):
                if not timestep_respacing.startswith("ddim"):
                    raise ValueError("Only 'ddimN' timestep respacing is supported")
                desired = int(timestep_respacing[4:])
            else:
                desired = int(timestep_respacing)
            stride = next(
                (
                    candidate
                    for candidate in range(1, base_num_timesteps)
                    if len(range(0, base_num_timesteps, candidate)) == desired
                ),
                None,
            )
            if stride is None:
                raise ValueError(
                    f"Cannot create exactly {desired} DDIM steps from "
                    f"{base_num_timesteps} base steps"
                )
            timestep_map = torch.arange(
                0, base_num_timesteps, stride, dtype=torch.long
            )
            base_alpha_bars = torch.cumprod(1.0 - base_betas, dim=0)
            last_alpha_bar = torch.tensor(1.0, dtype=torch.float64)
            spaced_betas = []
            for timestep in timestep_map.tolist():
                alpha_bar = base_alpha_bars[timestep]
                spaced_betas.append(1.0 - alpha_bar / last_alpha_bar)
                last_alpha_bar = alpha_bar
            betas = torch.stack(spaced_betas)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.num_timesteps = len(betas)
        self.base_num_timesteps = base_num_timesteps
        self.schedule = schedule
        self.register_buffer("timestep_map", timestep_map)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt().float())
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt().float()
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape) * noise
        )

    def random_timesteps(self, batch_size: int, device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def model_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.timestep_map[timesteps.long()]

    def _sampling_grid(self, steps: int, *, descending: bool) -> list[int]:
        if steps < 2:
            raise ValueError("DDIM steps must be at least 2")
        grid = torch.linspace(0, self.num_timesteps - 1, steps).round().long().unique()
        values = grid.tolist()
        return list(reversed(values)) if descending else values

    @torch.no_grad()
    def ddim_sample(
        self,
        model: Callable,
        latent: torch.Tensor,
        *,
        steps: int = 50,
        clip: tuple[float, float] | None = (0.0, 1.0),
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        model_kwargs = model_kwargs or {}
        x = latent
        grid = self._sampling_grid(steps, descending=True)
        for position, timestep in enumerate(grid):
            t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
            predicted_x0 = model(x, self.model_timesteps(t), **model_kwargs)
            if clip is not None:
                predicted_x0 = predicted_x0.clamp(*clip)
            alpha = self.alpha_bars[timestep]
            epsilon = (x - alpha.sqrt() * predicted_x0) / (1.0 - alpha).sqrt().clamp_min(1e-8)
            previous = grid[position + 1] if position + 1 < len(grid) else -1
            if previous < 0:
                x = predicted_x0
            else:
                alpha_previous = self.alpha_bars[previous]
                x = (
                    alpha_previous.sqrt() * predicted_x0
                    + (1.0 - alpha_previous).sqrt() * epsilon
                )
        return x

    @torch.no_grad()
    def ddpm_sample(
        self,
        model: Callable,
        latent: torch.Tensor,
        *,
        clip: tuple[float, float] | None = (0.0, 1.0),
        model_kwargs: dict | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Ancestral x0 sampling over every step, with fixed-small posterior variance.

        A separate generator keeps reverse-process noise independent of the
        initial-latent RNG used by the population/chunk sampler.
        """
        model_kwargs = model_kwargs or {}
        x = latent
        for timestep in reversed(range(self.num_timesteps)):
            t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
            x0 = model(x, self.model_timesteps(t), **model_kwargs)
            if clip is not None:
                x0 = x0.clamp(*clip)
            if timestep == 0:
                return x0
            beta = self.betas[timestep]
            alpha_bar = self.alpha_bars[timestep]
            previous_bar = self.alpha_bars[timestep - 1]
            denominator = 1.0 - alpha_bar
            mean = (
                previous_bar.sqrt() * beta / denominator * x0
                + (1.0 - beta).sqrt() * (1.0 - previous_bar) / denominator * x
            )
            variance = beta * (1.0 - previous_bar) / denominator
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            x = mean + variance.clamp_min(0.0).sqrt() * noise
        return x

    @torch.no_grad()
    def ddim_invert(
        self,
        model: Callable,
        clean: torch.Tensor,
        *,
        steps: int = 50,
        clip: tuple[float, float] | None = (0.0, 1.0),
        model_kwargs: dict | None = None,
    ) -> torch.Tensor:
        model_kwargs = model_kwargs or {}
        x = clean
        grid = self._sampling_grid(steps, descending=False)
        for position, timestep in enumerate(grid[:-1]):
            next_timestep = grid[position + 1]
            t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
            predicted_x0 = model(x, self.model_timesteps(t), **model_kwargs)
            if clip is not None:
                predicted_x0 = predicted_x0.clamp(*clip)
            alpha = self.alpha_bars[timestep]
            epsilon = (x - alpha.sqrt() * predicted_x0) / (1.0 - alpha).sqrt().clamp_min(1e-8)
            alpha_next = self.alpha_bars[next_timestep]
            x = alpha_next.sqrt() * predicted_x0 + (1.0 - alpha_next).sqrt() * epsilon
        return x
