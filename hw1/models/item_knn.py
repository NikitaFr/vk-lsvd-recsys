"""Self-implemented item cosine kNN. No IO, relevance labelling or ranking."""
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy import sparse


K_GRID = (5, 10, 20, 40, 80, 160, 320, 640, 1000, 1500, 1999)
SIGNALS = ("binary", "graded")
AGGREGATIONS = ("sum", "normalized")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool/float")
    return int(value)


def _ids(values, name, *, unique=True):
    # object conversion preserves mixed signed/uint64 Python integers exactly;
    # never let NumPy first infer float64 for such a mixed list.
    a = np.asarray(values, dtype=object)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = [_integer(v, name) for v in a]
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return np.array(result, dtype=object)


class ItemKNN:
    """Cosine neighborhoods of candidates, with numeric-ID tie breaking.

    Public array properties return defensive copies. A failed fit invalidates
    previous fitted state. Signal is immutable; create a new model to change it.
    """

    def __init__(self, *, signal: str = "graded") -> None:
        if not isinstance(signal, str) or signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}")
        self._signal = signal
        self._state = None

    @property
    def signal(self):
        return self._signal

    def _fitted(self):
        if self._state is None:
            raise ValueError("fit must succeed before accessing model or scores")
        return self._state

    @property
    def user_ids(self):
        return self._fitted()[0].copy()

    @property
    def item_ids(self):
        return self._fitted()[1].copy()

    @property
    def training_matrix(self):
        return self._fitted()[2].copy()

    @property
    def similarity(self):
        return self._fitted()[3].copy()

    @property
    def neighbors(self):
        """Column positions, shape (items, items-1); use item_ids to decode."""
        return self._fitted()[4].copy()

    def fit(self, history: pd.DataFrame, user_ids, item_ids, *,
            history_start_week: int = 0, history_end_week: int = 24) -> "ItemKNN":
        self._state = None
        start = _integer(history_start_week, "history_start_week")
        end = _integer(history_end_week, "history_end_week")
        if start < 0 or end < start:
            raise ValueError("history week bounds must satisfy 0 <= start <= end")
        users = _ids(user_ids, "user_ids")
        items = _ids(item_ids, "item_ids")
        if len(users) < 1 or len(items) < 2:
            raise ValueError("at least one user and two items are required")
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        required = {"user_id", "item_id", "week", "rel"}
        if not history.columns.is_unique or not required.issubset(history.columns):
            raise ValueError("history needs unique columns user_id, item_id, week, rel")
        hu = _ids(history["user_id"], "history.user_id", unique=False)
        hi = _ids(history["item_id"], "history.item_id", unique=False)
        weeks = [_integer(w, "history.week") for w in history["week"]]
        if any(w < start or w > end for w in weeks):
            raise ValueError("history.week outside declared history bounds")
        rel = history["rel"].tolist()
        if any(isinstance(r, (bool, np.bool_)) or not isinstance(r, Real)
               or not np.isfinite(r) or r not in (0, 1, 2) for r in rel):
            raise ValueError("history.rel must contain finite numeric 0, 1 or 2")
        if history.duplicated(["user_id", "item_id"]).any():
            raise ValueError("duplicate (user_id, item_id) history pairs")
        umap = {uid: i for i, uid in enumerate(users)}
        imap = {iid: i for i, iid in enumerate(items)}
        try:
            rows = np.fromiter((umap[u] for u in hu), dtype=np.int64, count=len(hu))
            cols = np.fromiter((imap[i] for i in hi), dtype=np.int64, count=len(hi))
        except KeyError as exc:
            raise ValueError(f"history contains unknown user/item ID: {exc.args[0]}") from exc
        values = np.asarray(rel, dtype=np.float64)
        if self.signal == "binary":
            values = (values >= 1).astype(np.float64)
        x = sparse.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
        x.eliminate_zeros()
        norms = np.sqrt(np.asarray(x.power(2).sum(axis=0)).ravel())
        # Normalize the Gram matrix by column norms (algebraically identical
        # to multiplying L2-normalized columns). Integer-valued dot products
        # remain exact before the single division, avoiding accumulation of
        # per-entry normalization roundoff at a neighborhood tie boundary.
        similarity = (x.T @ x).toarray()
        norm_products = norms[:, None] * norms[None, :]
        np.divide(similarity, norm_products, out=similarity, where=norm_products > 0)
        np.clip(similarity, 0.0, 1.0, out=similarity)
        np.fill_diagonal(similarity, 0.0)
        # Stable sort after numeric-ID ordering: ties follow ID, not column index.
        numeric_order = np.array(sorted(range(len(items)), key=lambda j: items[j]))
        neighbors = np.empty((len(items), len(items) - 1), dtype=np.int64)
        for i in range(len(items)):
            others = numeric_order[numeric_order != i]
            neighbors[i] = others[np.argsort(-similarity[i, others], kind="stable")]
        self._state = (users, items, x, similarity, neighbors)
        return self

    def _numerator(self, k):
        _, items, x, similarity, neighbors = self._fitted()
        k = _integer(k, "k")
        if not 1 <= k < len(items):
            raise ValueError(f"k must be between 1 and {len(items)-1}")
        selected = neighbors[:, :k]
        weights = similarity[np.arange(len(items))[:, None], selected]
        denominator = weights.sum(axis=1)
        # W[i,j] = similarity(candidate i, neighbor j). Scores = X @ W.T.
        w = sparse.csr_matrix((weights.ravel(), selected.ravel(),
                               np.arange(len(items) + 1) * k),
                              shape=(len(items), len(items)))
        w.eliminate_zeros()
        numerator = (x @ w.T).toarray()
        return numerator, denominator

    def score(self, *, k: int, aggregation: str = "sum") -> np.ndarray:
        if not isinstance(aggregation, str) or aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        numerator, denominator = self._numerator(k)
        if aggregation == "normalized":
            np.divide(numerator, denominator, out=numerator, where=denominator > 0)
        return numerator

    def score_both(self, *, k: int) -> dict[str, np.ndarray]:
        numerator, denominator = self._numerator(k)
        normalized = np.divide(numerator, denominator,
                               out=np.zeros_like(numerator), where=denominator > 0)
        return {"sum": numerator, "normalized": normalized}
