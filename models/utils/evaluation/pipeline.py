"""Pure evaluator: explicit history, explicit holdout, no IO or model fitting."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
import pandas as pd
from ._validation import IDInput, columns, ids, pairs, positive_int, real_array, relevance
from .ranking import rank_scores
from .metrics import user_metrics, aggregate_metrics


@dataclass(frozen=True)
class EvaluationResult:
    """Recommendations, unchanged-NaN personal metrics, JSON-safe macro summary."""
    recommendations: pd.DataFrame
    per_user: pd.DataFrame
    summary: dict


def evaluate_scores(scores: np.ndarray, user_ids: IDInput, item_ids: IDInput, history: pd.DataFrame,
                    holdout: pd.DataFrame, k: int = 10, *,
                    history_start_week: int = 0, history_end_week: int = 24,
                    holdout_week: int = 25) -> EvaluationResult:
    """Evaluate the full fixed catalog using the supplied axis order.

    Defaults enforce the validation protocol. Explicit temporal arguments allow
    future reuse without adding file loaders. History requires integer-valued
    week; holdout week is optional, but must equal holdout_week when provided.
    Without it the caller attests to holdout origin. Overlaps are always errors.
    Neither model training provenance nor truth of week labels can be inferred.
    """
    k = positive_int(k, 'k')
    for value in (history_start_week, history_end_week, holdout_week):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            raise ValueError('week boundaries must be nonnegative integers')
    if not history_start_week <= history_end_week < holdout_week:
        raise ValueError('require history_start_week <= history_end_week < holdout_week')
    users, items = ids(user_ids, 'user_ids'), ids(item_ids, 'item_ids')
    columns(history, ('user_id', 'item_id', 'week'), 'history')
    weeks = real_array(history['week'], 'history.week')
    if ((weeks != np.floor(weeks)) | (weeks < history_start_week) | (weeks > history_end_week)).any():
        raise ValueError('history.week outside allowed integer week boundaries')
    # Ranking has no holdout argument: eligibility/feedback cannot affect Top-K.
    recommendations = rank_scores(scores, users, items, history, k)
    columns(holdout, ('user_id', 'item_id', 'rel'), 'holdout')
    hu, hi = pairs(holdout, users, items, 'holdout')
    rel = relevance(holdout['rel'])
    if 'week' in holdout:
        w = real_array(holdout['week'], 'holdout.week')
        if (w != holdout_week).any():
            raise ValueError('holdout.week differs from expected holdout_week')
    known = pd.MultiIndex.from_frame(history[['user_id', 'item_id']])
    future = pd.MultiIndex.from_frame(holdout[['user_id', 'item_id']])
    if future.isin(known).any():
        raise ValueError('holdout overlaps history user-item pairs')
    truth = {u: {} for u in users}
    for u, i, r in zip(hu, hi, rel):
        truth[u][i] = int(r)
    rec_items = recommendations['item_id'].to_numpy().reshape(len(users), k)
    rows = [user_metrics([truth[u].get(i, 0) for i in rec_items[j]],
                         list(truth[u].values()), k) for j, u in enumerate(users)]
    per_user = pd.DataFrame(rows)
    per_user.insert(0, 'user_id', users)
    summary = aggregate_metrics(per_user, k)
    summary.update({'n_items': len(items), 'n_history': len(history), 'n_holdout': len(holdout),
                    'history_start_week': int(history_start_week),
                    'history_end_week': int(history_end_week), 'holdout_week': int(holdout_week),
                    'observed_history_min_week': int(weeks.min()) if len(weeks) else None,
                    'observed_history_max_week': int(weeks.max()) if len(weeks) else None,
                    'holdout_week_checked': 'week' in holdout})
    return EvaluationResult(recommendations, per_user, summary)
