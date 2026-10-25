"""User-level percentile bootstrap of fixed personal metrics; no retraining."""
import numpy as np
import pandas as pd
from ._validation import positive_int
from .metrics import validate_metric_table


def _select(e: np.ndarray, a: np.ndarray, group: str) -> np.ndarray:
    if group == 'eligible':
        if not e.any():
            raise ValueError('no eligible users for bootstrap')
        return a[e]
    if group == 'all_users':
        return np.where(e, a, 0)
    raise ValueError("group must be 'eligible' or 'all_users'")


def _interval(values: np.ndarray, n_bootstrap: int, ci: float, seed: int) -> dict:
    n_bootstrap = positive_int(n_bootstrap, 'n_bootstrap')
    if isinstance(ci, (bool, np.bool_)) or not isinstance(ci, (int, float, np.integer, np.floating)) or not 0 < ci < 1:
        raise ValueError('ci must be a finite number strictly between 0 and 1')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        means[b] = values[rng.integers(0, len(values), size=len(values))].mean()
    alpha = (1 - ci) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha], method='linear')
    return {'estimate': float(values.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'ci': float(ci), 'n_bootstrap': n_bootstrap, 'seed': int(seed), 'n_users': len(values),
            'method': 'percentile; linear quantiles; resampling users'}


def bootstrap_mean(per_user: pd.DataFrame, metric: str = 'ndcg_at_10', *,
                   group: str = 'eligible', n_bootstrap: int = 2000,
                   ci: float = .95, seed: int = 42) -> dict:
    """Bootstrap a macro mean. Sort original user IDs for row-order invariance.

    All-users replaces personal NaN by zero in a temporary array only. CI is
    conditional on this offline user sample; exposure bias/MNAR are not cured.
    """
    users, eligible, values = validate_metric_table(per_user, metric)
    order = np.argsort(users, kind='stable')
    selected = _select(eligible[order], values[order], group)
    return {**_interval(selected, n_bootstrap, ci, seed), 'metric': metric, 'group': group}


def bootstrap_paired_difference(model_a: pd.DataFrame, model_b: pd.DataFrame,
                                metric: str = 'ndcg_at_10', *, group: str = 'eligible',
                                n_bootstrap: int = 2000, ci: float = .95, seed: int = 42) -> dict:
    """Percentile CI for A-B; exact ID alignment and identical user resamples.

    Resampling the vector of aligned personal differences is algebraically
    identical to resampling both models with the same index vector.
    """
    ua, ea, a = validate_metric_table(model_a, metric)
    ub, eb, b = validate_metric_table(model_b, metric)
    if set(ua) != set(ub):
        raise ValueError('models must have exactly the same user IDs')
    oa, ob = np.argsort(ua, kind='stable'), np.argsort(ub, kind='stable')
    if not np.array_equal(ea[oa], eb[ob]):
        raise ValueError('models have different eligible masks for matched user IDs')
    if 'n_relevant' in model_a and 'n_relevant' in model_b:
        if not np.array_equal(model_a['n_relevant'].to_numpy()[oa], model_b['n_relevant'].to_numpy()[ob]):
            raise ValueError('models have different n_relevant for matched user IDs')
    diff = _select(ea[oa], a[oa], group) - _select(eb[ob], b[ob], group)
    return {**_interval(diff, n_bootstrap, ci, seed), 'metric': metric, 'group': group,
            'difference': 'model_A - model_B'}
