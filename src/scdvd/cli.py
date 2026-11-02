"""Installed command entry points; each stage accepts Hydra overrides."""
from __future__ import annotations

import sys


def main() -> None:
    commands = {"train", "train-mask", "evaluate"}
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: scdvd {train,train-mask,evaluate} [Hydra options and key=value overrides]")
        print("Example: scdvd train --config-name sciplex3 --cfg job")
        return
    command = sys.argv.pop(1)
    if command == "--version":
        from scdvd import __version__
        print(__version__)
        return
    if command not in commands:
        raise SystemExit(f"Unknown command {command!r}; choose from {', '.join(sorted(commands))}")
    if command == "train":
        from scdvd.train import main as run
    elif command == "train-mask":
        from scdvd.train_mask import main as run
    else:
        from scdvd.evaluate import main as run
    run()
