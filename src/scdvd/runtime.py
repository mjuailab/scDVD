"""Small, explicit provenance records for training and evaluation."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from omegaconf import OmegaConf


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def metadata_dict(metadata):
    return asdict(metadata)


def record_run(directory, config, *, stage, metadata=None, artifacts=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, directory / "resolved_config.yaml", resolve=True)
    versions = {}
    for name in ("torch", "lightning", "numpy", "h5py", "geomloss", "hydra-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {"stage": stage, "python": platform.python_version(), "packages": versions}
    if metadata is not None:
        record["dataset_metadata"] = metadata_dict(metadata)
    if artifacts:
        record["artifacts"] = {
            name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            for name, path in artifacts.items() if path is not None
        }
    (directory / "run_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
