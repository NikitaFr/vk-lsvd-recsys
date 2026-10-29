"""Analytical EASE; no file I/O, ranking, masking, or holdout access."""
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

__all__ = ["EASE", "PreparedEASEData", "prepare_ease_data"]
_TOKEN = object()


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must contain integers (not bool/float/string)")
    return int(value)


def _ids(values, name, *, unique=False):
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        raise TypeError(f"{name} must be a one-dimensional ID sequence")
    if getattr(values, "ndim", 1) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # Iterate original scalars: np.asarray([large_uint64, small_int]) may round.
    result = [_integer(v, name) for v in values]
    if any(v < 0 for v in result):
        raise ValueError(f"{name} IDs must be nonnegative")
    if unique and (not result or len(set(result)) != len(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    # Object dtype preserves even Python integers beyond uint64 without coercion.
    return np.array(result, dtype=object)


def _signal(signal):
    if not isinstance(signal, str) or signal not in ("binary", "graded"):
        raise ValueError("signal must be 'binary' or 'graded'")
    return signal


def _bounds(start, end):
    start, end = _integer(start, "history_start_week"), _integer(end, "history_end_week")
    if not 0 <= start <= end:
        raise ValueError("history bounds must satisfy 0 <= start <= end")
    return start, end


class PreparedEASEData:
    """Validated snapshot. Public array properties return independent copies.

    Construct exclusively through prepare_ease_data. Private attributes are not
    an API; ordinary public assignment is blocked. No Gram recomputation at fit.
    """
    __slots__ = ("_X", "_G", "_users", "_items", "_signal", "_start", "_end", "_validated")

    def __init__(self, token=None, *, X=None, G=None, users=None, items=None,
                 signal=None, start=None, end=None):
        if token is not _TOKEN:
            raise TypeError("Use prepare_ease_data() to create validated data")
        for key, value in (("_X", X), ("_G", G), ("_users", users), ("_items", items)):
            value.setflags(write=False)
            object.__setattr__(self, key, value)
        for key, value in (("_signal", signal), ("_start", start), ("_end", end),
                           ("_validated", True)):
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("PreparedEASEData is immutable; prepare a new snapshot")

    X = property(lambda self: self._X.copy())
    G = property(lambda self: self._G.copy())
    user_ids = property(lambda self: self._users.copy())
    item_ids = property(lambda self: self._items.copy())
    signal = property(lambda self: self._signal)
    history_start_week = property(lambda self: self._start)
    history_end_week = property(lambda self: self._end)


def prepare_ease_data(history, user_ids, item_ids, *, signal="binary",
                      history_start_week, history_end_week):
    """Validate all observations and build float64 X and Gram once per signal.

    Explicit bounds are caller-owned: use 0..24 for validation experiments,
    0..25 only for the separately authorized final refit. Never filters rows.
    """
    signal = _signal(signal)
    start, end = _bounds(history_start_week, history_end_week)
    users, items = _ids(user_ids, "user_ids", unique=True), _ids(item_ids, "item_ids", unique=True)
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")
    required = {"user_id", "item_id", "week", "rel"}
    if not history.columns.is_unique or not required.issubset(history.columns):
        raise ValueError("history requires unique columns user_id, item_id, week, rel")
    hu, hi = _ids(history.user_id, "history.user_id"), _ids(history.item_id, "history.item_id")
    weeks = [_integer(v, "week") for v in history.week]
    if any(w < start or w > end for w in weeks):
        raise ValueError(f"week outside authorized history range [{start}, {end}]")
    rel = []
    for v in history.rel:
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                or not np.isfinite(v) or v not in (0, 1, 2)):
            raise ValueError("rel must contain finite integer-valued 0, 1, 2")
        rel.append(float(v))
    umap, imap = {v: i for i, v in enumerate(users)}, {v: i for i, v in enumerate(items)}
    try:
        rows = np.fromiter((umap[v] for v in hu), dtype=np.intp, count=len(hu))
        cols = np.fromiter((imap[v] for v in hi), dtype=np.intp, count=len(hi))
    except KeyError as exc:
        raise ValueError(f"history contains unknown ID {exc.args[0]}") from exc
    pairs = rows * len(items) + cols
    if len(np.unique(pairs)) != len(pairs):
        raise ValueError("duplicate (user_id, item_id) in history")
    X = np.zeros((len(users), len(items)), dtype=np.float64)
    values = np.asarray(rel, dtype=np.float64)
    X[rows, cols] = (values >= 1) if signal == "binary" else values
    G = X.T @ X
    if not np.isfinite(G).all():
        raise FloatingPointError("non-finite Gram matrix")
    return PreparedEASEData(_TOKEN, X=X, G=G, users=users, items=items,
                            signal=signal, start=start, end=end)


class EASE:
    """Minimize .5||X-XB||² + .5 lambda ||B||² subject to diag(B)=0.

    Fits are atomic: failed refits preserve the last successful fitted state.
    Constructor hyperparameters are read-only. No random seed is required.
    """
    def __init__(self, regularization, signal="binary"):
        if (isinstance(regularization, (bool, np.bool_))
                or not isinstance(regularization, Real)):
            raise ValueError("regularization must be a finite real number > 0")
        try:
            lam = float(regularization)
        except (OverflowError, ValueError) as exc:
            raise ValueError("regularization must be finite") from exc
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("regularization must be finite and > 0")
        self._regularization = lam
        self._signal = _signal(signal)
        self._state = None

    regularization = property(lambda self: self._regularization)
    signal = property(lambda self: self._signal)

    def fit(self, history, user_ids, item_ids, *, history_start_week, history_end_week):
        prepared = prepare_ease_data(history, user_ids, item_ids, signal=self.signal,
                                     history_start_week=history_start_week,
                                     history_end_week=history_end_week)
        return self.fit_prepared(prepared)

    def fit_prepared(self, prepared):
        if not isinstance(prepared, PreparedEASEData) or not prepared._validated:
            raise TypeError("prepared must be validated PreparedEASEData")
        if prepared.signal != self.signal:
            raise ValueError("prepared.signal does not match model.signal")
        _bounds(prepared.history_start_week, prepared.history_end_week)
        X, G = prepared._X, prepared._G
        n = len(prepared._items)
        if (X.shape != (len(prepared._users), n) or G.shape != (n, n)
                or X.dtype != np.float64 or G.dtype != np.float64
                or not np.isfinite(X).all() or not np.isfinite(G).all()):
            raise ValueError("invalid prepared shapes, dtypes, or non-finite values")
        A = G.copy()
        A.flat[::n + 1] += self.regularization
        try:
            P = cho_solve(cho_factor(A, lower=True, overwrite_a=True,
                                    check_finite=True), np.eye(n), check_finite=True)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise np.linalg.LinAlgError("EASE SPD solve failed; inspect input scale and lambda") from exc
        diagonal = P.diagonal().copy()
        if not np.isfinite(P).all() or not (diagonal > 0).all():
            raise FloatingPointError("EASE precision matrix must be finite with positive diagonal")
        # Divide each TARGET column j by P[j,j]. Reuse P storage for B.
        P /= -diagonal[None, :]
        np.fill_diagonal(P, 0.0)
        if not np.isfinite(P).all():
            raise FloatingPointError("non-finite EASE coefficients")
        P.setflags(write=False)
        self._state = (prepared, P)
        return self

    def _fitted(self):
        if self._state is None:
            raise RuntimeError("EASE is not fitted; call fit or fit_prepared first")
        return self._state

    @property
    def coef_(self):
        return self._fitted()[1].copy()

    @property
    def user_ids_(self):
        return self._fitted()[0].user_ids

    @property
    def item_ids_(self):
        return self._fitted()[0].item_ids

    def score(self):
        prepared, B = self._fitted()
        scores = prepared._X @ B
        if not np.isfinite(scores).all():
            raise FloatingPointError("non-finite EASE scores")
        return scores
