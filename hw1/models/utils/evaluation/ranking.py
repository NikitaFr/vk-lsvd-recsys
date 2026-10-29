"""Full-catalog ranking independent of holdout feedback."""
import numpy as np
import pandas as pd
from ._validation import IDInput, ids, pairs, positive_int, real_array


def rank_scores(scores: np.ndarray, user_ids: IDInput, item_ids: IDInput, history: pd.DataFrame,
                k: int = 10) -> pd.DataFrame:
    """Return K unseen items per user in explicit input user order.

    Scores correspond exactly to the input axes; neither axis is silently
    sorted. Ties use ORIGINAL numeric item ID ascending. All historical pairs
    are masked regardless of reactions. Temporal checks belong to the pipeline.
    """
    k = positive_int(k, 'k')
    users, items = ids(user_ids, 'user_ids'), ids(item_ids, 'item_ids')
    s = real_array(scores, 'scores')
    if s.ndim != 2 or s.shape != (len(users), len(items)):
        raise ValueError('scores shape must be (len(user_ids), len(item_ids))')
    hu, hi = pairs(history, users, items, 'history')
    umap, imap = {u: j for j, u in enumerate(users)}, {i: j for j, i in enumerate(items)}
    seen = np.zeros(s.shape, dtype=bool)
    seen[[umap[u] for u in hu], [imap[i] for i in hi]] = True
    counts = len(items) - seen.sum(axis=1)
    if (counts < k).any():
        raise ValueError(f'fewer than k={k} candidates for user IDs {users[counts < k].tolist()}')
    top = np.empty((len(users), k), dtype=np.intp)
    descending_ids = np.argsort(items, kind='stable')[::-1]
    for row in range(len(users)):
        candidates = descending_ids[~seen[row, descending_ids]]
        # Reverse stable ascending score order on ID-desc input. This also
        # handles signed/unsigned integer score extrema without negation overflow.
        order = np.argsort(s[row, candidates], kind='stable')[::-1]
        top[row] = candidates[order[:k]]
    return pd.DataFrame({
        'user_id': np.repeat(users, k), 'rank': np.tile(np.arange(1, k + 1), len(users)),
        'item_id': items[top.ravel()], 'score': s[np.arange(len(users))[:, None], top].ravel(),
    })
