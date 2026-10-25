"""Exact weighted implicit ALS; no data access or ranking/evaluation policy."""
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer, not bool')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be real, not bool')
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _ids(values, name, unique=False):
    # Object conversion BEFORE inference prevents mixed signed/uint64 lists
    # from being silently rounded through float64 by NumPy.
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    vals = [_integer(v, name) for v in raw]
    if unique and (not vals or len(set(vals)) != len(vals)):
        raise ValueError(f'{name} must be nonempty and unique')
    if all(0 <= v <= np.iinfo(np.uint64).max for v in vals):
        return np.array(vals, dtype=np.uint64)
    if all(np.iinfo(np.int64).min <= v <= np.iinfo(np.int64).max for v in vals):
        return np.array(vals, dtype=np.int64)
    return np.array(vals, dtype=object)


def _values(values, name, length, lower, upper=None):
    raw = np.asarray(values, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) for v in raw.flat):
        raise TypeError(f'{name} must not contain bool values')
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != length:
        raise ValueError(f'{name} must be 1D with length {length}')
    if arr.dtype.kind not in 'iuf':
        raise TypeError(f'{name} must have a real numeric dtype (not bool/object)')
    arr = arr.astype(np.float64, copy=True)
    if not np.isfinite(arr).all() or (arr < lower).any() or (upper is not None and (arr > upper).any()):
        raise ValueError(f'{name} contains nonfinite or out-of-range values')
    return arr


def _targets_confidences(rel, alpha, mode, targets=None, confidences=None):
    t = (rel >= 1).astype(np.float64) if targets is None else _values(targets, 'observed_targets', len(rel), 0, 1)
    with np.errstate(over='raise', invalid='raise'):
        c = (1. + alpha * ((rel >= 1) if mode == 'binary' else rel)) if confidences is None else _values(confidences, 'observed_confidences', len(rel), 1)
    if not np.isfinite(c).all():
        raise ValueError('generated confidences are not finite')
    return t, c


def _index(rows, columns, targets, confidences, size):
    order = np.argsort(rows, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(rows, minlength=size))]
    return offsets, columns[order], targets[order], confidences[order]


def _solve_row(fixed, gram, indices, targets, confidences, regularization):
    """Solve one normal equation; gram includes ALL opposite-side factors."""
    selected = fixed[indices]
    matrix = gram + selected.T @ ((confidences - 1)[:, None] * selected)
    matrix = matrix.copy()
    matrix.flat[::matrix.shape[0] + 1] += regularization
    rhs = selected.T @ (confidences * targets)
    answer = cho_solve(cho_factor(matrix, lower=True, check_finite=True), rhs, check_finite=True)
    if not np.isfinite(answer).all():
        raise FloatingPointError('ALS produced nonfinite factors')
    return answer


def _update(fixed, indexed, regularization):
    offsets, columns, targets, confidences = indexed
    gram = fixed.T @ fixed
    result = np.empty((len(offsets) - 1, fixed.shape[1]), dtype=np.float64)
    for row in range(len(result)):
        sl = slice(offsets[row], offsets[row + 1])
        result[row] = _solve_row(fixed, gram, columns[sl], targets[sl], confidences[sl], regularization)
    return result


def _loss(a, b, rows, columns, targets, confidences, regularization):
    """Exact objective, with bounded observed-pair working memory."""
    value = np.sum((a.T @ a) * (b.T @ b))
    for start in range(0, len(rows), 8192):
        sl = slice(start, start + 8192)
        s = np.einsum('ij,ij->i', a[rows[sl]], b[columns[sl]])
        t, c = targets[sl], confidences[sl]
        value += np.sum((c - 1) * s**2 - 2*c*t*s + c*t**2)
    value += regularization * (np.sum(a*a) + np.sum(b*b))
    if not np.isfinite(value):
        raise FloatingPointError('ALS objective is not finite')
    return float(value)


class IALS:
    """Model #2 with independent optional observed targets/confidences.

    Parameters are read-only. Failed fit invalidates the previous fit; failed
    continuation retains the last complete iteration. All exposed arrays/logs
    are copies. Private underscore attributes are implementation details.
    """

    def __init__(self, factors=32, regularization=10.0, alpha=40.0,
                 confidence_mode='binary', random_seed=42):
        self._factors = _integer(factors, 'factors', 1)
        self._regularization = _positive(regularization, 'regularization')
        self._alpha = _positive(alpha, 'alpha')
        if not isinstance(confidence_mode, str) or confidence_mode not in ('binary', 'graded'):
            raise ValueError('confidence_mode must be binary or graded')
        self._confidence_mode = confidence_mode
        self._random_seed = _integer(random_seed, 'random_seed')
        self._state = None

    factors = property(lambda self: self._factors)
    regularization = property(lambda self: self._regularization)
    alpha = property(lambda self: self._alpha)
    confidence_mode = property(lambda self: self._confidence_mode)
    random_seed = property(lambda self: self._random_seed)

    def _ready(self):
        if self._state is None:
            raise RuntimeError('IALS requires a successful fit first')
        return self._state

    user_ids = property(lambda self: self._ready()['users'].copy())
    item_ids = property(lambda self: self._ready()['items'].copy())
    user_factors = property(lambda self: self._ready()['a'].copy())
    item_factors = property(lambda self: self._ready()['b'].copy())
    n_iterations = property(lambda self: len(self._ready()['log']))
    training_log = property(lambda self: [entry.copy() for entry in self._ready()['log']])

    def fit(self, history, user_ids, item_ids, n_iters=5,
            observed_targets=None, observed_confidences=None,
            history_start_week=0, history_end_week=24):
        """Fit from scratch, validating the complete declared historical window."""
        self._state = None
        n_iters = _integer(n_iters, 'n_iters', 1)
        start = _integer(history_start_week, 'history_start_week', 0)
        end = _integer(history_end_week, 'history_end_week', 0)
        if start > end:
            raise ValueError('history_start_week exceeds history_end_week')
        users, items = _ids(user_ids, 'user_ids', True), _ids(item_ids, 'item_ids', True)
        if not isinstance(history, pd.DataFrame):
            raise TypeError('history must be a pandas DataFrame')
        if not history.columns.is_unique or not {'user_id', 'item_id', 'week', 'rel'} <= set(history.columns):
            raise ValueError('history requires unique columns user_id, item_id, week, rel')
        hu, hi = _ids(history['user_id'], 'history.user_id'), _ids(history['item_id'], 'history.item_id')
        weeks = [_integer(w, 'history.week', 0) for w in history['week']]
        if any(w < start or w > end for w in weeks):
            raise ValueError('history contains weeks outside the declared window')
        # Empty frames may have object dtype for rel; there are no values to reject.
        rel = _values(history['rel'] if len(history) else [], 'rel', len(history), 0, 2)
        if not np.isin(rel, [0, 1, 2]).all():
            raise ValueError('rel must contain only 0, 1, 2')
        um, im = {int(v): j for j, v in enumerate(users)}, {int(v): j for j, v in enumerate(items)}
        try:
            rows = np.array([um[int(v)] for v in hu], dtype=np.intp)
            cols = np.array([im[int(v)] for v in hi], dtype=np.intp)
        except KeyError as exc:
            raise ValueError(f'history ID outside fixed axes: {exc.args[0]}') from exc
        pairs = pd.MultiIndex.from_arrays([rows, cols])
        if pairs.has_duplicates:
            raise ValueError('duplicate user-item pair in history')
        t, c = _targets_confidences(rel, self.alpha, self.confidence_mode, observed_targets, observed_confidences)
        # default_rng rejects negative seeds; deterministic unsigned residue also
        # supports the contract's unrestricted integer seeds.
        seed = self.random_seed if self.random_seed >= 0 else self.random_seed % (2**64)
        state = dict(users=users, items=items, rows=rows, cols=cols, t=t, c=c,
                     ui=_index(rows, cols, t, c, len(users)),
                     ii=_index(cols, rows, t, c, len(items)),
                     a=np.zeros((len(users), self.factors)),
                     b=np.random.default_rng(seed).normal(0., .01, (len(items), self.factors)), log=[])
        # Publish only after ALL requested iterations succeed.
        self._iterate(state, n_iters)
        self._state = state
        return self

    def _iterate(self, state, count):
        for _ in range(count):
            begin = perf_counter()
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                a = _update(state['b'], state['ui'], self.regularization)
                b = _update(a, state['ii'], self.regularization)
                loss = _loss(a, b, state['rows'], state['cols'], state['t'], state['c'], self.regularization)
            state['a'], state['b'] = a, b
            state['log'].append(dict(iteration=len(state['log']) + 1, loss=loss,
                                     elapsed_seconds=perf_counter() - begin))

    def continue_fit(self, additional_iterations=5):
        state = self._ready()
        count = _integer(additional_iterations, 'additional_iterations', 1)
        self._iterate(state, count)
        return self

    def score(self):
        state = self._ready()
        with np.errstate(over='raise', invalid='raise'):
            scores = state['a'] @ state['b'].T
        if not np.isfinite(scores).all():
            raise FloatingPointError('scores are not finite')
        return scores
