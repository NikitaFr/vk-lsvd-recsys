"""Shared strict validation; never coerce identifiers through floating point."""
from numbers import Integral
from typing import Sequence
import numpy as np
import pandas as pd


IDInput = np.ndarray | pd.Series | pd.Index | Sequence[int]

def positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def ids(values: IDInput, name: str, *, unique: bool = True, nonempty: bool = True) -> np.ndarray:
    # Object construction prevents NumPy from inferring float for mixed int64/uint64.
    a = values.to_numpy(copy=True) if isinstance(values, (pd.Series, pd.Index)) else (
        values.copy() if isinstance(values, np.ndarray) else np.array(values, dtype=object))
    if a.ndim != 1 or (nonempty and len(a) == 0):
        raise ValueError(f'{name} must be a nonempty 1D integer ID array')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in a):
        raise ValueError(f'{name} must contain integer IDs without missing values; floats forbidden')
    if unique and len(set(a)) != len(a):
        raise ValueError(f'{name} contains duplicate IDs')
    return a


def columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f'{name} must be a DataFrame with unique columns')
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')


def real_array(values, name: str) -> np.ndarray:
    a = np.asarray(values)
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite real numeric values (no bool/object/complex)')
    return a


def relevance(values, name: str = 'rel') -> np.ndarray:
    a = real_array(values, name)
    if a.ndim != 1 or not np.isin(a, [0, 1, 2]).all():
        raise ValueError(f'{name} must be 1D with values in {{0,1,2}}')
    return a.astype(np.int8)


def pairs(frame: pd.DataFrame, users: np.ndarray, items: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    columns(frame, ('user_id', 'item_id'), name)
    u = ids(frame['user_id'], f'{name}.user_id', unique=False, nonempty=False)
    i = ids(frame['item_id'], f'{name}.item_id', unique=False, nonempty=False)
    if not set(u).issubset(set(users)) or not set(i).issubset(set(items)):
        raise ValueError(f'{name} contains unknown user/item IDs')
    if frame.duplicated(['user_id', 'item_id']).any():
        raise ValueError(f'{name} contains duplicate user-item pairs')
    return u, i
