"""Second-order FM on observed events. No I/O on import or data preparation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ENCODER_VERSION = 1
FEATURE_SETS = {
    'id': ('user_id', 'item_id'),
    'item_meta': ('user_id', 'item_id', 'author_id', 'duration'),
    'user_meta': ('user_id', 'item_id', 'author_id', 'duration', 'age', 'gender'),
    'full': ('user_id', 'item_id', 'author_id', 'duration', 'age', 'gender', 'geo'),
}


def _integer(x, name, minimum=None):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral):
        raise ValueError(f'{name} must be an integer (no floats/strings)')
    x = int(x)
    if minimum is not None and x < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return x


def _ids(values, name, unique=False):
    result = tuple(_integer(v, name, 0) for v in values)
    if unique and (not result or len(set(result)) != len(result)):
        raise ValueError(f'{name}: empty or duplicate IDs')
    return result


def _categories(values, name):
    return tuple(_integer(v, name) for v in values)


def _readonly(a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    # Immutable bytes backing: cannot re-enable WRITEABLE with setflags.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _table(table, key, axis, columns):
    if not isinstance(table, pd.DataFrame) or not table.columns.is_unique:
        raise ValueError('expected DataFrame with unique column names')
    if not {key, *columns}.issubset(table.columns):
        raise ValueError(f'metadata missing required columns: {key}, {columns}')
    keys = _ids(table[key], key, unique=True)
    if set(keys) != set(axis):
        raise ValueError(f'{key}: metadata must cover exactly the fixed axis')
    positions = {v: i for i, v in enumerate(keys)}
    return table.iloc[[positions[v] for v in axis]]


def _numeric(series, name):
    values = list(series)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError(f'{name} must be numeric')
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or (name == 'duration' and (a <= 0).any()):
        raise ValueError(f'{name}: nonfinite or invalid value')
    mean, std = float(a.mean()), float(a.std(ddof=0))
    if not np.isfinite([mean, std]).all():
        raise ValueError(f'{name}: invalid scaler')
    z = np.zeros_like(a) if std == 0 else (a - mean) / std
    return z.astype(np.float32), (name, mean, std)


@dataclass(frozen=True)
class FMSnapshot:
    user_ids: tuple
    item_ids: tuple
    feature_set: str
    fields: tuple
    vocabularies: tuple
    scalers: tuple
    n_features: int
    user_indices: np.ndarray
    user_values: np.ndarray
    item_indices: np.ndarray
    item_values: np.ndarray
    version: int = ENCODER_VERSION

    def encode_positions(self, users, items):
        # Each side stores zeros in fields owned by the other side.
        return (self.user_indices[users] + self.item_indices[items],
                self.user_values[users] + self.item_values[items])


@dataclass(frozen=True)
class PreparedFMData:
    encoder: FMSnapshot
    indices: np.ndarray
    values: np.ndarray
    labels: np.ndarray
    history_start_week: int
    history_end_week: int
    history_rows: tuple

    @property
    def fingerprint(self):
        e = self.encoder
        h = hashlib.sha256(json.dumps({
            'version': e.version, 'users': e.user_ids, 'items': e.item_ids,
            'feature_set': e.feature_set, 'fields': e.fields,
            'vocab': e.vocabularies, 'scalers': e.scalers, 'D': e.n_features,
            'period': [self.history_start_week, self.history_end_week],
            'history': self.history_rows,
        }, separators=(',', ':'), allow_nan=False).encode())
        for a in (e.user_indices, e.user_values, e.item_indices, e.item_values,
                  self.indices, self.values, self.labels):
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        return h.hexdigest()


def prepare_fm_data(*, history, users_metadata, items_metadata, user_ids,
                    item_ids, feature_set='full', history_start_week,
                    history_end_week):
    """Validate all supplied metadata; encode only approved active fields.

    IDs must be nonnegative integers; categorical codes must be integers. Metadata snapshots
    must contain all seven source fields even for an ablation. Extra columns
    are ignored. No period is inferred and no rows are silently removed.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError('unknown feature_set')
    start = _integer(history_start_week, 'history_start_week', 0)
    end = _integer(history_end_week, 'history_end_week', start)
    users, items = _ids(user_ids, 'user_ids', True), _ids(item_ids, 'item_ids', True)
    um = _table(users_metadata, 'user_id', users, ('age', 'gender', 'geo'))
    im = _table(items_metadata, 'item_id', items, ('author_id', 'duration'))
    age, age_stats = _numeric(um.age, 'age')
    duration, duration_stats = _numeric(im.duration, 'duration')
    source = {'user_id': users, 'item_id': items,
              'gender': _categories(um.gender, 'gender'), 'geo': _categories(um.geo, 'geo'),
              'author_id': _categories(im.author_id, 'author_id'),
              'age': age, 'duration': duration}
    fields = FEATURE_SETS[feature_set]
    ui = np.zeros((len(users), len(fields)), np.int64)
    uv = np.zeros(ui.shape, np.float32)
    ii = np.zeros((len(items), len(fields)), np.int64)
    iv = np.zeros(ii.shape, np.float32)
    offset, vocab, stats = 0, [], []
    for j, name in enumerate(fields):
        ix, val = (ui, uv) if name in ('user_id', 'age', 'gender', 'geo') else (ii, iv)
        if name in ('age', 'duration'):
            ix[:, j], val[:, j] = offset, source[name]
            stats.append(age_stats if name == 'age' else duration_stats)
            offset += 1
        else:
            codes = tuple(sorted(set(source[name])))
            mapping = {code: offset + k for k, code in enumerate(codes)}
            ix[:, j] = [mapping[code] for code in source[name]]
            val[:, j] = 1
            vocab.append((name, offset, codes))
            offset += len(codes)
    encoder = FMSnapshot(users, items, feature_set, fields, tuple(vocab), tuple(stats),
                         offset, _readonly(ui, np.int64), _readonly(uv, np.float32),
                         _readonly(ii, np.int64), _readonly(iv, np.float32))
    if (not isinstance(history, pd.DataFrame) or not history.columns.is_unique or
            not {'user_id', 'item_id', 'week', 'rel'}.issubset(history.columns) or len(history) == 0):
        raise ValueError('history must be a nonempty DataFrame with user_id/item_id/week/rel')
    hu, hi = _ids(history.user_id, 'history user_id'), _ids(history.item_id, 'history item_id')
    weeks = tuple(_integer(x, 'week') for x in history.week)
    rels = tuple(_integer(x, 'rel') for x in history.rel)
    if any(w < start or w > end for w in weeks):
        raise ValueError('history contains events outside the explicitly allowed weeks')
    if any(r not in (0, 1, 2) for r in rels):
        raise ValueError('rel must be 0, 1, or 2')
    if len(set(zip(hu, hi))) != len(hu):
        raise ValueError('duplicate observed user/item pairs')
    u_map, i_map = {x: k for k, x in enumerate(users)}, {x: k for k, x in enumerate(items)}
    try:
        up, ip = [u_map[x] for x in hu], [i_map[x] for x in hi]
    except KeyError as exc:
        raise ValueError('unknown history ID') from exc
    indices, values = encoder.encode_positions(up, ip)
    return PreparedFMData(encoder, _readonly(indices, np.int64),
                          _readonly(values, np.float32),
                          _readonly(np.asarray(rels) >= 1, np.float32), start, end,
                          tuple(zip(hu, hi, weeks, rels)))


class FMCore(nn.Module):
    """FM identity: b + sum(w*x) + .5 sum((sum(V*x))² - sum((V*x)²))."""
    def __init__(self, n_features, n_factors, seed=42):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Parameter(torch.zeros(n_features))
        self.factors = nn.Parameter(torch.randn(n_features, n_factors, generator=generator) * .01)

    def forward(self, indices, values):
        vx = self.factors[indices] * values.unsqueeze(-1)
        interactions = .5 * (vx.sum(dim=1).square() - vx.square().sum(dim=1)).sum(dim=1)
        return self.bias + (self.linear[indices] * values).sum(dim=1) + interactions


class FactorizationMachine:
    def __init__(self, n_factors=16, learning_rate=.001, weight_decay=.01,
                 batch_size=8192, seed=42, device='cpu', verbose=False,
                 progress_callback: Callable | None = None):
        self.n_factors = _integer(n_factors, 'n_factors', 1)
        self.batch_size = _integer(batch_size, 'batch_size', 1)
        self.seed = _integer(seed, 'seed', 0)
        for name, value in [('learning_rate', learning_rate), ('weight_decay', weight_decay)]:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 < learning_rate <= 1 or weight_decay < 0:
            raise ValueError('require 0 < learning_rate <= 1 and weight_decay >= 0')
        self.learning_rate, self.weight_decay = float(learning_rate), float(weight_decay)
        if device not in ('cpu', 'mps'):
            raise ValueError('device must be cpu or mps')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable; explicitly select device="cpu"')
        self.device, self.verbose, self.progress_callback = device, verbose, progress_callback
        self.trained_epochs, self.training_log = 0, []
        self.prepared = None
        self.last_score_seconds = None

    def _config(self):
        return {k: getattr(self, k) for k in ('n_factors', 'learning_rate', 'weight_decay', 'batch_size', 'seed')}

    def _ready(self):
        if self.prepared is None:
            raise RuntimeError('model is not fitted; call fit_prepared first')
        if self.prepared is not self._bound_prepared:
            raise ValueError('prepared history/encoder changed; call fit_prepared for a new fit')
        if self.device != self._fit_device:
            raise ValueError('device changed after fit; reload checkpoint on the target device')
        if self._config() != self._fit_config:
            raise ValueError('configuration changed after fit; create or fit a new model')

    @property
    def user_ids(self):
        self._ready()
        return self.prepared.encoder.user_ids

    @property
    def item_ids(self):
        self._ready()
        return self.prepared.encoder.item_ids

    @property
    def feature_set(self):
        self._ready()
        return self.prepared.encoder.feature_set

    def _initialize(self, prepared):
        if not isinstance(prepared, PreparedFMData):
            raise TypeError('prepared must come from prepare_fm_data')
        self.prepared = prepared
        self._bound_prepared = prepared
        self._fit_device = self.device
        self._fit_config = self._config()
        self._fingerprint = prepared.fingerprint
        self.core = FMCore(prepared.encoder.n_features, self.n_factors, self.seed).to(self.device)
        self.optimizer = torch.optim.AdamW([
            {'params': [self.core.linear, self.core.factors], 'weight_decay': self.weight_decay},
            {'params': [self.core.bias], 'weight_decay': 0.0},
        ], lr=self.learning_rate)
        self._rng = torch.Generator(device='cpu').manual_seed(self.seed)
        # CPU buffers; only current mini-batch is transferred to MPS.
        self._indices = torch.tensor(prepared.indices, dtype=torch.long)
        self._values = torch.tensor(prepared.values, dtype=torch.float32)
        self._labels = torch.tensor(prepared.labels, dtype=torch.float32)
        self.trained_epochs, self.training_log = 0, []

    def fit_prepared(self, prepared, epochs=5):
        epochs = _integer(epochs, 'epochs', 1)
        self._initialize(prepared)
        return self.continue_fit(epochs)

    def fit(self, *, epochs=5, **data):
        return self.fit_prepared(prepare_fm_data(**data), epochs)

    def _sync(self):
        if self.device == 'mps':
            torch.mps.synchronize()

    def continue_fit(self, additional_epochs=5):
        self._ready()
        count = _integer(additional_epochs, 'additional_epochs', 1)
        n = len(self._labels)
        self.core.train()
        for _ in range(count):
            self._sync()
            start = time.perf_counter()
            permutation = torch.randperm(n, generator=self._rng)
            loss_sum, seen = 0., 0
            for begin in range(0, n, self.batch_size):
                positions = permutation[begin:begin + self.batch_size]
                indices = self._indices[positions].to(self.device)
                values = self._values[positions].to(self.device)
                labels = self._labels[positions].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(self.core(indices, values), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite train BCE')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.core.parameters()):
                    raise FloatingPointError('nonfinite gradients')
                self.optimizer.step()
                loss_sum += loss.detach().item() * len(positions)
                seen += len(positions)
            self._sync()
            if any(not torch.isfinite(p).all() for p in self.core.parameters()):
                raise FloatingPointError('nonfinite parameters')
            self.trained_epochs += 1
            row = dict(epoch=self.trained_epochs, train_bce=loss_sum / seen,
                       n_examples=seen, seconds=time.perf_counter() - start,
                       device=self.device, seed=self.seed, torch_version=str(torch.__version__))
            self.training_log.append(row)
            if self.verbose:
                print(row)
            if self.progress_callback:
                self.progress_callback(dict(row))
        return self

    @torch.inference_mode()
    def _predict_positions(self, users, items):
        indices, values = self.prepared.encoder.encode_positions(users, items)
        logits = self.core(torch.as_tensor(indices, device=self.device),
                           torch.as_tensor(values, device=self.device))
        result = logits.cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite scores')
        return result

    def score_pairs(self, user_ids, item_ids, batch_size=None):
        self._ready()
        users, items = _ids(user_ids, 'user_ids'), _ids(item_ids, 'item_ids')
        if len(users) != len(items):
            raise ValueError('pair axes must have equal length')
        batch = self.batch_size if batch_size is None else _integer(batch_size, 'batch_size', 1)
        um = {x: k for k, x in enumerate(self.user_ids)}
        im = {x: k for k, x in enumerate(self.item_ids)}
        try:
            up, ip = np.array([um[x] for x in users]), np.array([im[x] for x in items])
        except KeyError as exc:
            raise ValueError('unknown scoring ID') from exc
        out = np.empty(len(users), np.float32)
        for start in range(0, len(out), batch):
            out[start:start + batch] = self._predict_positions(up[start:start + batch], ip[start:start + batch])
        return out

    def score(self, batch_size=None):
        self._ready()
        batch = self.batch_size if batch_size is None else _integer(batch_size, 'batch_size', 1)
        self._sync()
        start = time.perf_counter()
        nu, ni = len(self.user_ids), len(self.item_ids)
        result = np.empty(nu * ni, np.float32)
        for begin in range(0, len(result), batch):
            positions = np.arange(begin, min(begin + batch, len(result)), dtype=np.int64)
            result[begin:begin + batch] = self._predict_positions(positions // ni, positions % ni)
        self._sync()
        self.last_score_seconds = time.perf_counter() - start
        return result.reshape(nu, ni)

    def _encoder_state(self):
        e = self.prepared.encoder
        metadata = json.dumps(dict(version=e.version, user_ids=e.user_ids,
            item_ids=e.item_ids, feature_set=e.feature_set, fields=e.fields,
            vocabularies=e.vocabularies, scalers=e.scalers, n_features=e.n_features),
            separators=(',', ':'), allow_nan=False)
        return dict(metadata=metadata,
                    user_indices=torch.tensor(e.user_indices), user_values=torch.tensor(e.user_values),
                    item_indices=torch.tensor(e.item_indices), item_values=torch.tensor(e.item_values))

    def save_checkpoint(self, path):
        """Save portable tensors/config. Reload requires the matching prepared snapshot."""
        self._ready()
        def cpu(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value
        payload = dict(version=1, config=self._fit_config, fingerprint=self._fingerprint,
                       encoder=self._encoder_state(),
                       model=cpu(self.core.state_dict()), optimizer=cpu(self.optimizer.state_dict()),
                       rng=self._rng.get_state(), trained_epochs=self.trained_epochs,
                       training_log=self.training_log)
        path = Path(path)
        temp = path.with_name(path.name + '.tmp')
        torch.save(payload, temp)
        temp.replace(path)

    @classmethod
    def load_checkpoint(cls, path, *, prepared, device='cpu', expected_config=None):
        """weights_only=True avoids arbitrary pickle execution; fingerprint binds data/config."""
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['version'] != 1 or payload['fingerprint'] != prepared.fingerprint:
            raise ValueError('checkpoint version/data/axes/encoder provenance mismatch')
        if expected_config is not None and payload['config'] != expected_config:
            raise ValueError('checkpoint configuration mismatch')
        model = cls(**payload['config'], device=device)
        model._initialize(prepared)
        snapshot = model._encoder_state()
        if (snapshot['metadata'] != payload['encoder']['metadata'] or
                any(not torch.equal(snapshot[k], payload['encoder'][k])
                    for k in ('user_indices', 'user_values', 'item_indices', 'item_values'))):
            raise ValueError('checkpoint encoder snapshot mismatch')
        model.core.load_state_dict(payload['model'], strict=True)
        model.optimizer.load_state_dict(payload['optimizer'])
        groups = model.optimizer.param_groups
        if (len(groups) != 2 or groups[0]['weight_decay'] != model.weight_decay or
                groups[1]['weight_decay'] != 0 or
                any(g['lr'] != model.learning_rate for g in groups)):
            raise ValueError('checkpoint optimizer configuration mismatch')
        for state in model.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('checkpoint optimizer has nonfinite state')
        model._rng.set_state(payload['rng'])
        model.trained_epochs = _integer(payload['trained_epochs'], 'trained_epochs', 0)
        model.training_log = payload['training_log']
        if len(model.training_log) != model.trained_epochs:
            raise ValueError('checkpoint epoch log mismatch')
        if any(not torch.isfinite(p).all() for p in model.core.parameters()):
            raise ValueError('checkpoint has nonfinite parameters')
        return model


def smoke_benchmark(prepared, factors=(16, 32), device='cpu'):
    """Explicit opt-in only: two epochs and full scoring per factor size."""
    rows = []
    for f in factors:
        model = FactorizationMachine(n_factors=f, device=device)
        begin = time.perf_counter()
        model.fit_prepared(prepared, epochs=2)
        fit_seconds = time.perf_counter() - begin
        scores = model.score()
        rows.append(dict(n_factors=f, fit_seconds=fit_seconds,
                         score_seconds=model.last_score_seconds, shape=scores.shape,
                         device=device, training_log=model.training_log))
    return rows
