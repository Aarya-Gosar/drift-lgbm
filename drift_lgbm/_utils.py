"""Small input-validation helpers shared across modules."""

from __future__ import annotations

from typing import Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def as_2d_float(X) -> Tuple[np.ndarray, Optional[List[Hashable]]]:
    """Convert ``X`` to a 2-D float64 array.

    Returns ``(array, column_names)`` where ``column_names`` is ``None`` unless
    ``X`` was a pandas object. A 1-D input is treated as a single feature.
    Only numeric (and boolean) columns are supported.
    """
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        non_numeric = [
            col for col, dtype in X.dtypes.items() if not pd.api.types.is_numeric_dtype(dtype)
        ]
        if non_numeric:
            raise TypeError(
                "drift-lgbm only supports numeric features; non-numeric columns: "
                f"{non_numeric[:10]}"
            )
        if not X.columns.is_unique:
            raise ValueError("DataFrame column names must be unique.")
        return X.to_numpy(dtype=np.float64, na_value=np.nan), list(X.columns)

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D input, got an array with shape {arr.shape}.")
    return arr, None


def align_columns(
    arr: np.ndarray,
    names: Optional[Sequence[Hashable]],
    expected_names: Sequence[Hashable],
    expected_have_names: bool,
) -> np.ndarray:
    """Check ``arr`` matches the expected feature layout, reordering by name if possible.

    If both the stored layout and the incoming data carry column names, the
    columns are matched by name (so a re-ordered DataFrame is fine). Otherwise
    columns are matched by position and only the count is checked.
    """
    n_expected = len(expected_names)
    if expected_have_names and names is not None:
        name_set, expected_set = set(names), set(expected_names)
        missing = [n for n in expected_names if n not in name_set]
        if missing:
            raise ValueError(f"Input is missing features seen during fit: {missing[:10]}")
        extra = [n for n in names if n not in expected_set]
        if extra:
            raise ValueError(f"Input has features not seen during fit: {extra[:10]}")
        if list(names) != list(expected_names):
            position = {n: i for i, n in enumerate(names)}
            arr = arr[:, [position[n] for n in expected_names]]
        return arr
    if arr.shape[1] != n_expected:
        raise ValueError(f"Expected {n_expected} features, got {arr.shape[1]}.")
    return arr


def as_1d(y, name: str = "y") -> np.ndarray:
    """Convert a target/weight vector to a 1-D numpy array."""
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr
