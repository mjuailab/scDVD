"""Compatibility entry point and historical analysis helpers."""
from scdvd.evaluate import (
    main, _condition_groups, _make_generation_batch, _matched_control_moments,
    _rank_condition_de_genes, _prefixed, _summary, _print_table,
)

if __name__ == "__main__":
    main()
