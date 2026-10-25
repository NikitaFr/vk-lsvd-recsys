"""Custom continuous-target iALS; shares the handwritten ALS solver with model 2.

Observed positive target: min(1, .4 + .6 log(1+min(t,60))/log(61)
                               + .1 I(like or bookmark)); otherwise zero.
The evaluation relevance and binary confidence remain unchanged.
"""
import numpy as np
from .ials import IALS


def time_only_target(t, rel, like, bookmark):
    """Historical name: duration component plus reaction bonus, gated by rel > 0."""
    t = np.asarray(t, dtype=float)
    rel = np.asarray(rel)
    like, bookmark = np.asarray(like), np.asarray(bookmark)
    if not (t.ndim == 1 and t.shape == rel.shape == like.shape == bookmark.shape):
        raise ValueError('Target inputs must be aligned one-dimensional arrays')
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('timespent must be finite and nonnegative')
    if not np.isin(rel, [0, 1, 2]).all():
        raise ValueError('rel must be 0, 1, or 2')
    if not np.isin(like, [0, 1]).all() or not np.isin(bookmark, [0, 1]).all():
        raise ValueError('Reactions must be boolean or 0/1')
    e = (like != 0) | (bookmark != 0)
    return np.where(rel > 0, np.minimum(1., .4 + .6 * np.log1p(np.minimum(t, 60.)) / np.log(61.) + .1 * e), 0.)


class CustomIALS(IALS):
    """Model 5: build its own target from historical events, then solve ALS."""
    def fit(self, history, user_ids, item_ids, n_iters=15,
            history_start_week=0, history_end_week=24):
        q = time_only_target(history['timespent'], history['rel'],
                             history['like'], history['bookmark'])
        return super().fit(history, user_ids, item_ids, n_iters=n_iters,
                           observed_targets=q, history_start_week=history_start_week,
                           history_end_week=history_end_week)
