"""Train-only uniform unknown-pair sampling for the reference observed-event FM."""
from __future__ import annotations

import copy
import math
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .fm import (prepare_fm_data, PreparedFMData, FMSnapshot, FMCore,
                 FactorizationMachine)

WNS_CHECKPOINT_VERSION = 2
SAMPLER_VERSION = 1
DEFAULT_MAX_CATALOG_PAIRS = 10_000_000


def _int(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _real(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be finite and >= 0 (no bool/strings)')
    return float(value)


def _count(ratio, n):
    product = _real(ratio, 'negative_ratio') * n
    if not math.isfinite(product):
        raise ValueError('negative_ratio * n_observed overflows')
    return math.floor(product + .5)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def weighted_batch_loss(logits, labels, weights, *, n_observed, n_sampled, negative_weight):
    """Uniform-shuffle estimator; last batch uses its actual length.

    Caller supplies weights 1 for observed rows and beta for sampled rows.
    This also accepts float64 tensors for independent numerical oracles.
    """
    beta = _real(negative_weight, 'negative_weight')
    n = _int(n_observed, 'n_observed', 1)
    m = _int(n_sampled, 'n_sampled')
    if logits.ndim != 1 or logits.numel() == 0 or logits.shape != labels.shape or logits.shape != weights.shape:
        raise ValueError('logits, labels, weights must be equal nonempty vectors')
    denominator = n + beta * m
    if not math.isfinite(denominator):
        raise ValueError('nonfinite weighted denominator')
    return (F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            * (weights * ((n + m) / denominator))).mean()


class UniformUnobservedSampler:
    """Exact uniform subset of all train-unobserved positional pairs.

    No RNG is owned here: sample(ratio, rng) advances the supplied Generator.
    Complement is built once; max_catalog_pairs bounds materialization.
    """
    def __init__(self, prepared: PreparedFMData, *, max_catalog_pairs=DEFAULT_MAX_CATALOG_PAIRS):
        if not isinstance(prepared, PreparedFMData):
            raise TypeError('prepared must come from prepare_fm_data')
        limit = _int(max_catalog_pairs, 'max_catalog_pairs', 1)
        e = prepared.encoder
        nu, ni = len(e.user_ids), len(e.item_ids)
        total = nu * ni  # Python integer: cannot wrap before the check.
        if not nu or not ni or total > np.iinfo(np.int64).max:
            raise ValueError('catalog cannot be represented by flat int64 positions')
        if total > limit:
            raise ValueError(f'catalog has {total} pairs, above max_catalog_pairs={limit}; increase only after checking RAM')
        um, im = {v: k for k, v in enumerate(e.user_ids)}, {v: k for k, v in enumerate(e.item_ids)}
        if len(um) != nu or len(im) != ni:
            raise ValueError('duplicate encoder axes')
        rows = prepared.history_rows
        n = len(rows)
        if not n or n != len(prepared.labels) or n != len(prepared.indices) or n != len(prepared.values):
            raise ValueError('inconsistent prepared history lengths')
        try:
            # One pass over already-Python history, never float-convert IDs.
            up = np.fromiter((um[row[0]] for row in rows), dtype=np.int64, count=n)
            ip = np.fromiter((im[row[1]] for row in rows), dtype=np.int64, count=n)
        except (KeyError, IndexError) as exc:
            raise ValueError('history IDs outside encoder axes') from exc
        known = up * ni + ip
        if len(np.unique(known)) != n:
            raise ValueError('duplicate observed history pairs')
        if any(len(row) != 4 or row[3] not in (0, 1, 2) or
               not prepared.history_start_week <= row[2] <= prepared.history_end_week for row in rows):
            raise ValueError('invalid train history rows/window')
        labels = np.fromiter((row[3] >= 1 for row in rows), dtype=np.float32, count=n)
        if not np.array_equal(labels, prepared.labels):
            raise ValueError('history/labels mismatch')
        # Check exact membership/encoding against the supplied prepared snapshot,
        # with bounded temporary feature buffers (not U*I*F).
        for begin in range(0, n, 65536):
            end = begin + 65536
            ix, val = e.encode_positions(up[begin:end], ip[begin:end])
            if not np.array_equal(ix, prepared.indices[begin:end]) or not np.array_equal(val, prepared.values[begin:end]):
                raise ValueError('history/encoder feature mismatch')
        # Boolean membership avoids setdiff sorting temporaries and arange(total).
        mask = np.ones(total, dtype=bool)
        mask[known] = False
        candidates = np.flatnonzero(mask)
        self.candidates = np.frombuffer(candidates.tobytes(), dtype=np.int64)
        self.n_observed, self.n_items, self.catalog_pairs = n, ni, total

    def sample(self, negative_ratio, rng):
        if not isinstance(rng, np.random.Generator):
            raise TypeError('rng must be numpy.random.Generator')
        count = _count(negative_ratio, self.n_observed)
        if count < 1:
            raise ValueError('positive sampling requires M >= 1')
        if count > len(self.candidates):
            raise ValueError(f'M={count} exceeds unknown capacity {len(self.candidates)}')
        return rng.choice(self.candidates, size=count, replace=False)


class SampledFactorizationMachine(FactorizationMachine):
    """Reference FM architecture with optional weighted synthetic target-zero pairs.

    Config is immutable after construction. Fresh fit resets training state;
    after a failed epoch fit again or reload the last completed checkpoint.
    """
    _CONFIG_KEYS = ('n_factors', 'learning_rate', 'weight_decay', 'batch_size', 'seed',
                    'negative_ratio', 'negative_weight', 'sampling_seed',
                    'sampling_strategy', 'max_catalog_pairs')

    def __setattr__(self, name, value):
        if getattr(self, '_config_locked', False) and name in self._CONFIG_KEYS:
            raise ValueError(f'{name} is immutable; create a new model for a new configuration')
        super().__setattr__(name, value)

    def __init__(self, n_factors=16, learning_rate=.001, weight_decay=.01,
                 batch_size=8192, seed=42, device='cpu', verbose=False,
                 progress_callback=None, *, negative_ratio=1., negative_weight=0.,
                 sampling_seed=None, sampling_strategy='uniform_unobserved',
                 max_catalog_pairs=DEFAULT_MAX_CATALOG_PAIRS):
        super().__init__(n_factors, learning_rate, weight_decay, batch_size,
                         seed, device, verbose, progress_callback)
        if self.seed >= 2**64:
            raise ValueError('seed must be < 2**64 for torch.Generator')
        self.negative_ratio = _real(negative_ratio, 'negative_ratio')
        self.negative_weight = _real(negative_weight, 'negative_weight')
        self.sampling_seed = self.seed + 1000 if sampling_seed is None else _int(sampling_seed, 'sampling_seed')
        if sampling_strategy != 'uniform_unobserved':
            raise ValueError('sampling_strategy must be uniform_unobserved')
        self.sampling_strategy = sampling_strategy
        self.max_catalog_pairs = _int(max_catalog_pairs, 'max_catalog_pairs', 1)
        if self.negative_weight > 0 and self.negative_ratio == 0:
            raise ValueError('positive negative_weight requires positive negative_ratio')
        self._config_locked = True
        self._failed_epoch = False

    def _config(self):
        return {k: getattr(self, k) for k in self._CONFIG_KEYS}

    def _ready(self):
        super()._ready()
        if self._failed_epoch:
            raise RuntimeError('incomplete epoch: fit again or reload a completed checkpoint')

    def _initialize(self, prepared):
        self._failed_epoch = True
        super()._initialize(prepared)
        self._sampling_rng = np.random.Generator(np.random.PCG64(self.sampling_seed))
        self._sampler = None
        self._n_sampled = 0
        if self.negative_weight > 0:
            self._sampler = UniformUnobservedSampler(prepared, max_catalog_pairs=self.max_catalog_pairs)
            self._n_sampled = _count(self.negative_ratio, len(prepared.labels))
            if not 1 <= self._n_sampled <= len(self._sampler.candidates):
                raise ValueError('requested M must be >=1 and <= unknown capacity')
        if not math.isfinite(len(prepared.labels) + self.negative_weight * self._n_sampled):
            raise ValueError('nonfinite weighted denominator')
        self._failed_epoch = False

    def _finish_fields(self, row, *, sampled_bce, sampling_seconds):
        n, m = len(self._labels), self._n_sampled
        row.update(observed_bce_preupdate=row['train_bce'], sampled_bce_preupdate=sampled_bce,
                   n_observed=n, n_sampled=m, negative_ratio=self.negative_ratio,
                   negative_weight=self.negative_weight, weighted_denominator=n + self.negative_weight * m,
                   n_optimizer_steps=math.ceil((n + m) / self.batch_size),
                   sampling_seconds=sampling_seconds, sampling_seed=self.sampling_seed)
        return row

    def continue_fit(self, additional_epochs=5):
        self._ready()
        count = _int(additional_epochs, 'additional_epochs', 1)
        for _ in range(count):
            if self.negative_weight == 0:
                # Delegate exactly: same shuffle, reductions, gradient/update order.
                callback, verbose = self.progress_callback, self.verbose
                self.progress_callback, self.verbose = None, False
                try:
                    super().continue_fit(1)
                except BaseException:
                    self._failed_epoch = True
                    raise
                finally:
                    self.progress_callback, self.verbose = callback, verbose
                row = self.training_log[-1]
                row['train_objective'] = row['train_bce']
                self._finish_fields(row, sampled_bce=None, sampling_seconds=0.)
            else:
                try:
                    row = self._weighted_epoch()
                except BaseException:
                    self._failed_epoch = True
                    raise
                self.trained_epochs += 1
                self.training_log.append(row)
            # User callback exceptions occur AFTER committing a complete epoch.
            if self.verbose:
                print(dict(row))
            if self.progress_callback:
                self.progress_callback(dict(row))
        return self

    def _weighted_epoch(self):
        self.core.train()
        self._sync()
        start = time.perf_counter()
        flat = self._sampler.sample(self.negative_ratio, self._sampling_rng)
        sampling_seconds = time.perf_counter() - start
        n, m = len(self._labels), len(flat)
        ni = self._sampler.n_items
        permutation = torch.randperm(n + m, generator=self._rng)
        obs_sum, neg_sum = 0., 0.
        for begin in range(0, n + m, self.batch_size):
            pos = permutation[begin:begin + self.batch_size]
            observed = pos < n
            # Assemble only the current batch, encoding sampled pairs with the
            # original snapshot. No extra epoch-size encoded feature matrix.
            ix = torch.empty((len(pos), self._indices.shape[1]), dtype=torch.long)
            val = torch.empty_like(ix, dtype=torch.float32)
            labels = torch.zeros(len(pos), dtype=torch.float32)
            ix[observed] = self._indices[pos[observed]]
            val[observed] = self._values[pos[observed]]
            labels[observed] = self._labels[pos[observed]]
            sampled_flat = flat[(pos[~observed] - n).numpy()]
            six, sval = self.prepared.encoder.encode_positions(sampled_flat // ni, sampled_flat % ni)
            ix[~observed], val[~observed] = torch.from_numpy(six), torch.from_numpy(sval)
            observed_device = observed.to(self.device)
            logits = self.core(ix.to(self.device), val.to(self.device))
            losses = F.binary_cross_entropy_with_logits(logits, labels.to(self.device), reduction='none')
            denominator = n + self.negative_weight * m
            # Normalize in Python float64 before float32 tensor construction:
            # finite beta may exceed float32 range, while these coefficients do not.
            scale = (n + m) / denominator
            weights = torch.where(observed_device, scale, self.negative_weight * scale)
            loss = (losses * weights).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite weighted objective')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.core.parameters()):
                raise FloatingPointError('nonfinite gradients')
            obs_sum += losses[observed_device].detach().sum().item()
            neg_sum += losses[~observed_device].detach().sum().item()
            self.optimizer.step()
        self._sync()
        if any(not torch.isfinite(p).all() for p in self.core.parameters()):
            raise FloatingPointError('nonfinite parameters')
        row = dict(epoch=self.trained_epochs + 1, train_bce=obs_sum / n,
                   train_objective=obs_sum / (n + self.negative_weight * m)
                       + (self.negative_weight / (n + self.negative_weight * m)) * neg_sum,
                   n_examples=n + m, seconds=time.perf_counter() - start,
                   device=self.device, seed=self.seed, torch_version=str(torch.__version__))
        return self._finish_fields(row, sampled_bce=neg_sum / m, sampling_seconds=sampling_seconds)

    def save_checkpoint(self, path):
        """Atomically replace path after successful save; create parent directories."""
        self._ready()
        payload = dict(version=WNS_CHECKPOINT_VERSION, model_type='fm_wns',
                       sampler_version=SAMPLER_VERSION, config=self._config(),
                       fingerprint=self._fingerprint, encoder=self._encoder_state(),
                       model=_cpu(self.core.state_dict()), optimizer=_cpu(self.optimizer.state_dict()),
                       rng=self._rng.get_state(), sampling_rng=copy.deepcopy(self._sampling_rng.bit_generator.state),
                       trained_epochs=self.trained_epochs, training_log=copy.deepcopy(self.training_log))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        try:
            torch.save(payload, temp)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path, *, prepared, device='cpu', expected_config=None):
        """Load only trusted-format tensors/primitives, always weights_only=True."""
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            required = {'version', 'model_type', 'sampler_version', 'config', 'fingerprint',
                        'encoder', 'model', 'optimizer', 'rng', 'sampling_rng', 'trained_epochs', 'training_log'}
            if not isinstance(payload, dict) or not required <= payload.keys():
                raise ValueError('checkpoint missing required fields')
            if (payload['version'] != WNS_CHECKPOINT_VERSION or payload['model_type'] != 'fm_wns'
                    or payload['sampler_version'] != SAMPLER_VERSION):
                raise ValueError('checkpoint model/version/sampler mismatch')
            if not isinstance(prepared, PreparedFMData) or payload['fingerprint'] != prepared.fingerprint:
                raise ValueError('checkpoint data/history/axes fingerprint mismatch')
            if set(payload['config']) != set(cls._CONFIG_KEYS):
                raise ValueError('checkpoint incomplete configuration')
            if expected_config is not None and payload['config'] != expected_config:
                raise ValueError('checkpoint configuration mismatch')
            model = cls(**payload['config'], device=device)
            model._initialize(prepared)
            snapshot = model._encoder_state()
            if snapshot['metadata'] != payload['encoder']['metadata'] or any(
                    not torch.equal(snapshot[k], payload['encoder'][k]) for k in snapshot if k != 'metadata'):
                raise ValueError('checkpoint encoder mismatch')
            current = model.core.state_dict()
            if set(payload['model']) != set(current) or any(
                    not torch.is_tensor(payload['model'][k]) or payload['model'][k].shape != v.shape
                    or payload['model'][k].dtype != v.dtype or not torch.isfinite(payload['model'][k]).all()
                    for k, v in current.items()):
                raise ValueError('checkpoint invalid model tensors')
            model.core.load_state_dict(payload['model'], strict=True)
            expected_groups = model.optimizer.state_dict()['param_groups']
            if payload['optimizer']['param_groups'] != expected_groups:
                raise ValueError('checkpoint optimizer groups/options mismatch')
            model.optimizer.load_state_dict(payload['optimizer'])
            groups = model.optimizer.param_groups
            if (len(groups) != 2 or groups[0]['weight_decay'] != model.weight_decay or groups[1]['weight_decay'] != 0
                    or any(g['lr'] != model.learning_rate or g['betas'] != (0.9, 0.999) or g['eps'] != 1e-8
                           or g['amsgrad'] or g['maximize'] for g in groups)):
                raise ValueError('checkpoint optimizer configuration mismatch')
            for parameter, state in model.optimizer.state.items():
                if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise ValueError('checkpoint invalid AdamW state fields')
                for key, value in state.items():
                    if not torch.is_tensor(value) or not torch.isfinite(value).all():
                        raise ValueError('checkpoint nonfinite/nonnumeric optimizer state')
                    if key == 'exp_avg_sq' and (value < 0).any():
                        raise ValueError('checkpoint negative second moment')
                    if key != 'step' and (value.shape != parameter.shape or value.dtype != parameter.dtype):
                        raise ValueError('checkpoint optimizer tensor mismatch')
            model._rng.set_state(payload['rng'])
            if payload['sampling_rng'].get('bit_generator') != 'PCG64':
                raise ValueError('checkpoint sampling RNG type mismatch')
            model._sampling_rng.bit_generator.state = payload['sampling_rng']
            model.trained_epochs = _int(payload['trained_epochs'], 'trained_epochs')
            model.training_log = payload['training_log']
            model._validate_log()
            expected_steps = sum(r['n_optimizer_steps'] for r in model.training_log)
            if expected_steps and len(model.optimizer.state) != 3:
                raise ValueError('checkpoint missing AdamW state')
            if any(s['step'].numel() != 1 or s['step'].item() != expected_steps for s in model.optimizer.state.values()):
                raise ValueError('checkpoint optimizer step mismatch')
            return model
        except (ValueError, TypeError, KeyError, RuntimeError, AttributeError, IndexError, OverflowError) as exc:
            raise ValueError(f'invalid WNS checkpoint: {exc}') from exc

    def _validate_log(self):
        if not isinstance(self.training_log, list) or len(self.training_log) != self.trained_epochs:
            raise ValueError('checkpoint epoch log mismatch')
        n, m = len(self._labels), self._n_sampled
        fixed = dict(n_observed=n, n_sampled=m, n_examples=n + m, negative_ratio=self.negative_ratio,
                     negative_weight=self.negative_weight, sampling_seed=self.sampling_seed, seed=self.seed,
                     weighted_denominator=n + self.negative_weight * m,
                     n_optimizer_steps=math.ceil((n + m) / self.batch_size))
        for epoch, row in enumerate(self.training_log, 1):
            if row['epoch'] != epoch or any(row[k] != v for k, v in fixed.items()):
                raise ValueError('checkpoint inconsistent epoch counters/configuration')
            for k in ('train_bce', 'train_objective', 'observed_bce_preupdate', 'seconds', 'sampling_seconds'):
                _real(row[k], f'log {k}')
            if row['train_bce'] != row['observed_bce_preupdate']:
                raise ValueError('checkpoint inconsistent observed loss')
            sampled = row['sampled_bce_preupdate']
            if m:
                _real(sampled, 'sampled_bce_preupdate')
            elif sampled is not None:
                raise ValueError('baseline must log sampled_bce_preupdate=None')
            objective = (n / fixed['weighted_denominator']) * row['train_bce']
            objective += (self.negative_weight * m / fixed['weighted_denominator']) * (sampled or 0.)
            if not math.isclose(objective, row['train_objective'], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError('checkpoint inconsistent objective')
            if row['device'] not in ('cpu', 'mps') or not isinstance(row['torch_version'], str):
                raise ValueError('checkpoint missing runtime provenance')
