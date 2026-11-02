"""Load trusted research checkpoints with relocatable molecular features.

The feature matrix is a nonpersistent buffer. Supply the same matrix (including
row order) at a new location; changing its contents changes the model.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import warnings

import torch

from .module import scDVDModule


def _load(path, model_type, *, map_location, chemical_feature_path=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    parameters = deepcopy(checkpoint["hyper_parameters"])
    if chemical_feature_path is not None:
        parameters["model_config"]["chemical_feature_path"] = str(chemical_feature_path)
    features = Path(parameters["model_config"]["chemical_feature_path"]).expanduser()
    if not features.is_file():
        raise FileNotFoundError(
            f"Molecular features not found: {features}. Supply chemical_feature_path "
            "pointing to the original feature matrix at its new location."
        )
    parameters["model_config"]["chemical_feature_path"] = str(features)
    state = dict(checkpoint["state_dict"])
    if model_type is scDVDModule:
        denoiser = parameters["model_config"]["denoiser"]
        # Pre-final checkpoints registered an unused gate even when disabled.
        # Only those unused keys may be removed; all other keys load strictly.
        if denoiser.get("population_gate_enabled", True) is False:
            unused = [k for k in state if k.startswith("denoiser.population_gate.")]
            if unused:
                warnings.warn("Removing unused population-gate parameters from a legacy checkpoint.")
                for key in unused:
                    state.pop(key)
    model = model_type(**parameters)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_diffusion_checkpoint(path: str | Path, *, map_location="cpu", chemical_feature_path=None):
    return _load(path, scDVDModule, map_location=map_location,
                 chemical_feature_path=chemical_feature_path)


def load_mask_checkpoint(path: str | Path, *, map_location="cpu", chemical_feature_path=None):
    from .mask_module import EmpiricalMaskModule
    return _load(path, EmpiricalMaskModule, map_location=map_location,
                 chemical_feature_path=chemical_feature_path)


def validate_dataset_metadata(model, metadata) -> None:
    """Reject different dimensions or known gene/category orderings."""
    expected = model.hparams["metadata"]
    for key in ("num_genes", "num_cell_types", "num_batches", "num_local_drugs"):
        if int(expected[key]) != int(getattr(metadata, key)):
            raise ValueError(f"Checkpoint/dataset mismatch: {key}")
    for key in ("gene_names", "gene_ids", "cell_types", "batches", "drugs"):
        if key in expected and tuple(expected[key]) != tuple(getattr(metadata, key)):
            raise ValueError(f"Checkpoint/dataset ordering mismatch: {key}")


def is_paper_model(model) -> bool:
    """Identify the paper's set4/gate-free/cap-free energy configuration."""
    config = model.hparams["model_config"]
    denoiser, objective = config["denoiser"], config["objective"]
    return (
        int(config["population"]["set_size"]) == 4
        and denoiser.get("population_gate_enabled", True) is False
        and denoiser.get("population_correction_rms_ratio", 0.5) is None
        and objective.get("distribution_loss") == "geomloss_energy"
        and float(objective.get("mmd_weight", 0)) == 0.001
        and objective.get("mask_padded_cells", False) is True
    )
