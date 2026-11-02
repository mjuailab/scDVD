from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def read_obs_column(group: h5py.Group, key: str) -> np.ndarray:
    """Read a dense or categorical AnnData obs column."""
    if key not in group:
        raise KeyError(f"obs column '{key}' is missing; available={list(group.keys())}")
    node = group[key]
    if isinstance(node, h5py.Group) and {"categories", "codes"}.issubset(node.keys()):
        categories = node["categories"][:]
        codes = node["codes"][:]
        values = categories[codes]
    elif isinstance(node, h5py.Dataset):
        values = node[:]
    else:
        raise TypeError(f"Unsupported H5AD obs representation for '{key}'")
    if values.dtype.kind in {"S", "O"}:
        values = np.asarray(
            [value.decode("utf-8") if isinstance(value, bytes) else value for value in values]
        )
    return values


class H5ExpressionMatrix:
    """Process-safe row reader for dense or CSR-encoded H5AD expression matrices."""

    def __init__(
        self,
        path: str | Path,
        scale: float = 1.0,
        *,
        clip_upper: bool = True,
    ) -> None:
        self.path = str(Path(path).resolve())
        self.scale = float(scale)
        self.clip_upper = bool(clip_upper)
        if self.scale <= 0:
            raise ValueError("normalization scale must be positive")
        self._file: h5py.File | None = None
        self._indptr: np.ndarray | None = None

        with h5py.File(self.path, "r") as handle:
            matrix = handle["X"]
            if isinstance(matrix, h5py.Group):
                required = {"data", "indices", "indptr"}
                if not required.issubset(matrix.keys()):
                    raise TypeError("Only CSR-encoded sparse H5AD matrices are supported")
                self.kind = "csr"
                self.shape = tuple(int(value) for value in matrix.attrs["shape"])
                self._indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
            else:
                self.kind = "dense"
                self.shape = tuple(int(value) for value in matrix.shape)

    def _matrix(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file["X"]

    def row(self, index: int) -> np.ndarray:
        matrix = self._matrix()
        if self.kind == "dense":
            row = np.asarray(matrix[index], dtype=np.float32)
        else:
            start = int(self._indptr[index])
            end = int(self._indptr[index + 1])
            row = np.zeros(self.shape[1], dtype=np.float32)
            if end > start:
                columns = np.asarray(matrix["indices"][start:end], dtype=np.int64)
                row[columns] = np.asarray(matrix["data"][start:end], dtype=np.float32)
        row = np.maximum(row / self.scale, 0.0)
        if self.clip_upper:
            row = np.minimum(row, 1.0)
        return row

    def rows(self, indices: Iterable[int]) -> np.ndarray:
        indices = list(int(index) for index in indices)
        output = np.empty((len(indices), self.shape[1]), dtype=np.float32)
        for position, index in enumerate(indices):
            output[position] = self.row(index)
        return output

    def row_block(self, start: int, stop: int) -> np.ndarray:
        """Read a contiguous row block with one HDF5 data access."""
        start, stop = int(start), int(stop)
        if not 0 <= start <= stop <= self.shape[0]:
            raise IndexError(f"Invalid row block [{start}:{stop}]")
        matrix = self._matrix()
        if self.kind == "dense":
            output = np.asarray(matrix[start:stop], dtype=np.float32)
        else:
            output = np.zeros((stop - start, self.shape[1]), dtype=np.float32)
            data_start = int(self._indptr[start])
            data_stop = int(self._indptr[stop])
            data = np.asarray(matrix["data"][data_start:data_stop], dtype=np.float32)
            columns = np.asarray(
                matrix["indices"][data_start:data_stop], dtype=np.int64
            )
            local_indptr = self._indptr[start : stop + 1] - data_start
            for local_row in range(stop - start):
                left = int(local_indptr[local_row])
                right = int(local_indptr[local_row + 1])
                output[local_row, columns[left:right]] = data[left:right]
        output = np.maximum(output / self.scale, 0.0)
        if self.clip_upper:
            output = np.minimum(output, 1.0)
        return output

    def maximum(self, indices: Iterable[int], chunk_size: int = 512) -> float:
        indices = np.asarray(list(int(index) for index in indices), dtype=np.int64)
        selected = np.zeros(self.shape[0], dtype=bool)
        selected[indices] = True
        maximum = 0.0
        matrix = self._matrix()
        row_chunk = max(int(chunk_size), 4096)
        if self.kind == "dense":
            for start in range(0, self.shape[0], row_chunk):
                stop = min(start + row_chunk, self.shape[0])
                local = selected[start:stop]
                if local.any():
                    values = np.asarray(matrix[start:stop], dtype=np.float32)
                    maximum = max(maximum, float(values[local].max()))
            return maximum

        indptr = self._indptr
        for start in range(0, self.shape[0], row_chunk):
            stop = min(start + row_chunk, self.shape[0])
            local_selected = selected[start:stop]
            if not local_selected.any():
                continue
            data_start = int(indptr[start])
            data_stop = int(indptr[stop])
            if data_stop <= data_start:
                continue
            values = np.asarray(
                matrix["data"][data_start:data_stop], dtype=np.float32
            )
            lengths = np.diff(indptr[start : stop + 1])
            row_ids = np.repeat(np.arange(stop - start), lengths)
            kept = local_selected[row_ids]
            if kept.any():
                maximum = max(maximum, float(values[kept].max()))
        return maximum

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except (AttributeError, TypeError):
                # h5py modules may already be partially torn down at interpreter exit.
                pass
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        self.close()
