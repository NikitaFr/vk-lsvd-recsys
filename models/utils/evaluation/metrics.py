"""Linear-gain ranking metrics and explicitly separated macro populations."""
from typing import Sequence
import numpy as np
import pandas as pd
from ._validation import columns, ids, positive_int, relevance, real_array


def user_metrics(recommended_rel: Sequence[int] | np.ndarray,
                 holdout_rel: Sequence[int] | np.ndarray, k: int = 10) -> dict[str, float | int | bool]:
    """Calculate metrics from K ranked grades and all candidate holdout grades.

    Unknown recommendations have grade 0 for scoring only. The caller must
    ensure unique recommended items and candidate-only ground truth. A grade
    multiplicity check rejects impossible positive hit counts.
    """
    k = positive_int(k, 'k')
    ranked = relevance(recommended_rel, 'recommended_rel')
    truth = relevance(holdout_rel, 'holdout_rel')
    if len(ranked) != k:
        raise ValueError('recommended_rel must contain exactly k grades')
    for grade in (1, 2):
        if np.sum(ranked == grade) > np.sum(truth == grade):
            raise ValueError('recommended positive grades are not supported by holdout')
    n = int(np.sum(truth > 0))
    result = {'n_relevant': n, 'eligible': n > 0}
    if n == 0:
        return {**result, f'ndcg_at_{k}': np.nan, f'recall_at_{k}': np.nan, f'ap_at_{k}': np.nan}
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(truth[truth > 0])[::-1][:k]
    idcg = np.dot(ideal, discount[:len(ideal)])
    hit = ranked > 0
    ap = np.sum(np.cumsum(hit) / np.arange(1, k + 1) * hit) / min(n, k)
    return {**result, f'ndcg_at_{k}': float(np.dot(ranked, discount) / idcg),
            f'recall_at_{k}': float(hit.sum() / n), f'ap_at_{k}': float(ap)}


def validate_metric_table(per_user: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns(per_user, ('user_id', 'eligible', metric), 'per_user')
    users = ids(per_user['user_id'], 'per_user.user_id')
    e = per_user['eligible'].to_numpy()
    if e.dtype.kind != 'b':
        raise ValueError('eligible must be nonmissing boolean values')
    a = per_user[metric].to_numpy()
    if a.dtype.kind not in 'iuf':
        raise ValueError('personal metrics must be numeric')
    if not np.isfinite(a[e]).all() or not np.isnan(a[~e]).all():
        raise ValueError('metrics must be finite for eligible users and NaN otherwise')
    if ((a[e] < 0) | (a[e] > 1 + 1e-12)).any():
        raise ValueError('personal ranking metrics must be in [0,1]')
    if 'n_relevant' in per_user:
        n = real_array(per_user['n_relevant'], 'n_relevant')
        if ((n < 0) | (n != np.floor(n))).any() or not np.array_equal(n > 0, e):
            raise ValueError('n_relevant must be nonnegative integral and agree with eligible')
    return users, e, a.astype(float)


def aggregate_metrics(per_user: pd.DataFrame, k: int = 10) -> dict:
    """Macro means; ineligible NaN becomes zero only in All-users aggregation.

    With no eligible users Eligible metrics are None (JSON null), All-users are
    zero. Personal NaNs are never modified. Bootstrap Eligible raises instead.
    """
    k = positive_int(k, 'k')
    eligible, all_users = {}, {}
    for metric, output in ((f'ndcg_at_{k}', f'ndcg_at_{k}'),
                           (f'recall_at_{k}', f'recall_at_{k}'), (f'ap_at_{k}', f'map_at_{k}')):
        users, e, a = validate_metric_table(per_user, metric)
        eligible[output] = float(a[e].mean()) if e.any() else None
        all_users[output] = float(np.where(e, a, 0).mean())
    return {'k': k, 'n_users': len(users), 'n_eligible': int(e.sum()),
            'eligible_share': float(e.mean()), 'eligible': eligible, 'all_users': all_users,
            'all_users_convention': 'Undefined personal metrics count as zero only for aggregation.',
            'gain': 'linear: 0,1,2'}
