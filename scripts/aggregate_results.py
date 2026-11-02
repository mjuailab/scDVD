"""Recreate scDVD table entries from per-run condition-average metrics.

Input columns: dataset, repeat, split, evaluation_split, genes, metric, value.
Each repeat must contain the five distinct splits; each setting three repeats.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics


def aggregate(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "evaluation_split", "genes", "metric"))
        run = (int(row["repeat"]), int(row["split"]))
        if run in groups[key]:
            raise ValueError(f"Duplicate run for {key}: {run}")
        value = float(row["value"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value for {key}: {run}")
        groups[key][run] = value
    if not groups:
        raise ValueError("No metrics to aggregate")
    output = []
    expected = {(r, s) for r in range(1, 4) for s in range(1, 6)}
    for key, runs in sorted(groups.items()):
        if set(runs) != expected:
            raise ValueError(f"Expected 3 repeats x 5 splits for {key}; missing={expected - set(runs)}")
        means = [statistics.mean(runs[r, s] for s in range(1, 6)) for r in range(1, 4)]
        output.append(dict(zip(("dataset", "evaluation_split", "genes", "metric"), key),
                           model="scDVD", set_size=4, mean=statistics.mean(means),
                           std=statistics.stdev(means), repeats=3, splits_per_repeat=5))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="") as handle:
        rows = aggregate(csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} table entries to {args.output}")


if __name__ == "__main__":
    main()
