"""Frozen VK-LSVD relevance rule (linear grades 0, 1, 2)."""
import numpy as np
import pandas as pd
from ._validation import columns, real_array


def label_relevance(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with int8 ``rel``; preserve index, IDs and all input columns.

    Requires timespent, duration, like, bookmark, dislike. Duration must already
    have been joined from the fixed item metadata by the caller. Ratio is not
    clipped; dislike has absolute precedence. No file access or implicit join.
    """
    columns(events, ('timespent', 'duration', 'like', 'bookmark', 'dislike'), 'events')
    t = real_array(events['timespent'], 'timespent')
    duration = real_array(events['duration'], 'duration')
    if (t < 0).any() or (duration <= 0).any():
        raise ValueError('timespent must be >=0 and duration >0')
    signals = []
    for name in ('like', 'bookmark', 'dislike'):
        a = events[name].to_numpy()
        if a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
            raise ValueError(f'{name} must contain only nonmissing binary 0/1 values')
        signals.append(a == 1)
    like, bookmark, dislike = signals
    # Overflow in the ratio for finite extreme values means very deep viewing,
    # not invalid input. All relevant comparisons still have their exact meaning.
    with np.errstate(over='ignore', under='ignore'):
        ratio = t / duration
    strong = ((ratio >= .80) & (t >= 5)) | ((like | bookmark) & (t >= 3))
    moderate = ((ratio >= .50) & (t >= 5)) | ((ratio >= .25) & (t >= 30))
    result = events.copy(deep=True)
    result['rel'] = np.select([dislike, strong, moderate], [0, 2, 1], default=0).astype(np.int8)
    return result
