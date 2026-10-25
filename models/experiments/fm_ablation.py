"""fm ablation experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

# Диагностика выполняется в ДОЧЕРНЕМ Python-процессе: нативный сбой не убьёт Jupyter.
import os
import platform
import signal
import subprocess
import sys
import textwrap

print('Jupyter Python executable:', sys.executable, flush=True)
print('Python:', sys.version.split()[0], '| platform:', platform.platform(),
      '| arch:', platform.machine(), flush=True)
print('Interpreter prefix:', sys.prefix, flush=True)

_CORE_PROBE = r"""
import sys
print('01 numpy', flush=True)
import numpy as np
print('   numpy:', np.__version__, flush=True)
print('02 pandas', flush=True)
import pandas as pd
print('   pandas:', pd.__version__, flush=True)
print('03 IPython.display', flush=True)
from IPython.display import display
print('04 torch', flush=True)
import torch
print('   torch:', torch.__version__, flush=True)
print('05 plotly', flush=True)
import plotly.express as px
import plotly.graph_objects as go
print('06 pyarrow', flush=True)
import pyarrow as pa
print('   pyarrow:', pa.__version__, flush=True)
print('PREFLIGHT_CORE_OK', flush=True)
"""

_MPS_PROBE = r"""
import torch
print('mps.is_built:', torch.backends.mps.is_built(), flush=True)
available = torch.backends.mps.is_available()
print('mps.is_available:', available, flush=True)
print('PREFLIGHT_MPS_AVAILABLE=' + str(int(available)), flush=True)
if available:
    x = torch.tensor([1.0], device='mps')
    print('mps test tensor:', (x * 2).cpu().item(), flush=True)
print('PREFLIGHT_MPS_OK', flush=True)
"""

def _probe_native_imports(title, script, timeout=90):
    try:
        proc = subprocess.run([sys.executable, '-X', 'faulthandler', '-u', '-c', script],
                              capture_output=True, text=True, timeout=timeout,
                              env=dict(os.environ))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f'{title}: дочерний Python завис (> {timeout} секунд). '
            f'Вывод до остановки: {exc.stdout!r} {exc.stderr!r}'
        ) from exc
    print('\n---', title, '---', flush=True)
    if proc.stdout:
        print(proc.stdout, end='' if proc.stdout.endswith('\n') else '\n', flush=True)
    if proc.stderr:
        print('stderr:', proc.stderr[-6000:], flush=True)
    if proc.returncode != 0:
        if proc.returncode < 0:
            try:
                reason = signal.Signals(-proc.returncode).name
            except (ValueError, AttributeError):
                reason = f'signal {-proc.returncode}'
            detail = f'аварийный сигнал {reason}'
        else:
            detail = f'exit code {proc.returncode}'
        return False, detail
    return True, 'OK'

FM_CORE_OK, fm_core_reason = _probe_native_imports('Импорты как в исходной первой ячейке', _CORE_PROBE)
if not FM_CORE_OK:
    raise RuntimeError(
        'Первый диагностический этап выявил проблему окружения: ' + fm_core_reason + '. '
        'Последняя строка перед остановкой показывает проблемный импорт. '
        'Проверьте совместимость библиотек и окружение: '
        'систему, архитектуру и версию Python, указанные выше.'
    )

FM_MPS_VALIDATED = False
if sys.platform == 'darwin':
    mps_ok, mps_reason = _probe_native_imports('Проверка MPS (отдельный процесс)', _MPS_PROBE)
    if mps_ok:
        # Убеждаемся, что GPU реально доступна и создала тензор, а не только импортировалась.
        # is_available может быть False даже при успешном завершении процесса.
        _mps = subprocess.run(
            [sys.executable, '-c',
             'import torch; print(int(torch.backends.mps.is_available()))'],
            capture_output=True, text=True, timeout=40
        )
        FM_MPS_VALIDATED = _mps.returncode == 0 and _mps.stdout.strip() == '1'
    else:
        print('MPS проверку не прошёл:', mps_reason, '\nОставляем только CPU.', flush=True)
else:
    print('MPS не применяется: не macOS.', flush=True)

print('PASS: основные зависимости успешно импортируются отдельно от Jupyter.', flush=True)
print('Проверенный режим FM:', 'mps' if FM_MPS_VALIDATED else 'cpu', flush=True)


from pathlib import Path
from itertools import product
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display

for package in ('pyarrow', 'torch', 'plotly', 'nbformat'):
    if importlib.util.find_spec(package) is None:
        raise ModuleNotFoundError(
            f'В kernel отсутствует {package}. Установи через %pip install {package}, '
            'перезапусти kernel и запусти Run All.'
        )
import torch
import plotly.express as px
import plotly.graph_objects as go


def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256_uint64_ids(values):
    arr = np.asarray(values)
    require(arr.ndim == 1 and arr.dtype.kind in 'iu' and not (arr < 0).any(),
            'ID должны быть одномерными неотрицательными целыми числами')
    return hashlib.sha256(np.sort(arr.astype(np.uint64)).tobytes()).hexdigest()


def write_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def root_of_project(start):
    for folder in (start, *start.parents):
        if (folder / 'data/work/config.json').is_file() and (folder / 'data/work/interactions.parquet').is_file():
            return folder
    raise FileNotFoundError('Не найден корень существующего VK-LSVD проекта (data/work/config.json и data/work).')

ROOT = root_of_project(Path.cwd().resolve())
FILES = {
    'config':ROOT/'data/work/config.json',
    'manifest':ROOT/'data/work/manifest.json',
    'revision':ROOT/'data/source/source_revision.json',
    'train':ROOT/'data/work/interactions.parquet',
    'users':ROOT/'data/work/users_metadata.parquet',
    'items':ROOT/'data/work/items_metadata.parquet',
    'validation':ROOT/'data/source/subsamples/up0.001_ip0.001/validation/week_25.parquet',
}
OUT = ROOT/'results/fm/wns_robustness_ablation_beta_1_5_10_20'
CACHE = OUT/'checkpoint_cache'
REVIEW = OUT/'review_bundle'
EXPECTED_FILE_SHA256 = {
    'train':'0c3a20a9729e30aeb650b28379d34663d3d2ec914bfe0ee3ca0a17d0904790a0',
    'users':'c0769c126e96499364f902362902690774005947d9fe23e097e43718d26d5cb2',
    'items':'3140e6749978a817e552a510c91f0c402eb93a639a096dc81c0bc99526b27d71',
    'validation':'d654b40a7f7e2086054c6131767967f84289fafd2364f566b984cc7d408917f5',
}
EXPECTED_ID_SHA256 = {
    'users':'ce51fd99e116ae5884714fcaf2244b6b99c4d5268ed961426ea86b8277c7780b',
    'items':'e69b0688d7d2ff4a68375a9bc711026d58899b565ce93c4c4c94692c29b59e76',
}
EXPECTED_FM_SHA256 = '83814ad4743be4aea653e8f44190e713b737b473c0be18b490158de01af1b7df'
EXPECTED_WNS_SHA256 = '8682fd4da2be0357fb6fe213cdcbb24a10a590711b1c095b126b626302b9dde4'
EXPECTED_EVAL_SHA256 = {
    'models/utils/evaluation/__init__.py':'e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c',
    'models/utils/evaluation/_validation.py':'6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405',
    'models/utils/evaluation/bootstrap.py':'9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221',
    'models/utils/evaluation/metrics.py':'14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938',
    'models/utils/evaluation/pipeline.py':'27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f',
    'models/utils/evaluation/ranking.py':'5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2',
    'models/utils/evaluation/relevance.py':'ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1',
}
# Узкая диагностическая сетка, НЕ подбираем другие гиперпараметры на этой итерации.
NEGATIVE_WEIGHTS = (1.0, 5.0, 10.0, 20.0)
SEEDS = (42, 43, 44)
ABLATION_FEATURE_SETS = ('id', 'item_meta', 'user_meta', 'full')
ABLATION_BETA = 5.0
ABLATION_SEED = 42
NEGATIVE_RATIO = 1.0
EPOCHS = (15, 25)
TOP_K = 10
N_FACTORS = 64
LEARNING_RATE = .003
WEIGHT_DECAY = .1
BATCH_SIZE = 8192
SEED = 42
SAMPLING_SEED = SEED + 1000  # use seed+1000 for every trajectory
BOOTSTRAP_N, BOOTSTRAP_SEED = 2000, 42
EXPERIMENT_DEVICE = 'mps' if FM_MPS_VALIDATED else 'cpu'
# По необходимости можно выбрать 'cpu' вручную до запуска тренировок.
RUN_TRAINING = True
RUN_PAIRWISE_BOOTSTRAP = True
REQUIRE_BASELINE_PARITY = False  # исторические данные доступны лишь для сравнения
print('Project:', ROOT)
print('PyTorch:', torch.__version__, '| device:', EXPERIMENT_DEVICE)
print('β:',NEGATIVE_WEIGHTS,'| seeds:',SEEDS,'| ablation:',ABLATION_FEATURE_SETS,'| epochs:',EPOCHS)


file_hashes = {key:sha256_file(FILES[key]) for key in EXPECTED_FILE_SHA256}
for key, expected in EXPECTED_FILE_SHA256.items():
    require(file_hashes[key] == expected, f'Изменён {key} Parquet: {file_hashes[key]}')
module_paths = ['models/fm.py', 'models/fm_wns.py', *EXPECTED_EVAL_SHA256.keys()]
module_hashes = {path:sha256_file(ROOT/path) for path in module_paths}
require(module_hashes['models/fm.py'] == EXPECTED_FM_SHA256,
        'Контрольная сумма models/fm.py отличается от ожидаемой')
require(module_hashes['models/fm_wns.py'] == EXPECTED_WNS_SHA256,
        'Контрольная сумма models/fm_wns.py отличается от ожидаемой')
for path, expected in EXPECTED_EVAL_SHA256.items():
    require(module_hashes[path] == expected, f'Изменён evaluator: {path}')
CFG = json.loads(FILES['config'].read_text(encoding='utf-8'))
MANIFEST = json.loads(FILES['manifest'].read_text(encoding='utf-8'))
REVISION = json.loads(FILES['revision'].read_text(encoding='utf-8'))
require(CFG['dataset_repo'] == MANIFEST['source'] == REVISION['repo'] == 'deepvk/VK-LSVD',
        'Изменился источник данных')
require(CFG['subsample'] == MANIFEST['original_slice'] == 'up0.001_ip0.001',
        'Изменён официальный slice')
require(REVISION['revision'] == MANIFEST['source_revision'] ==
        '084a18386e149a454dd6dbe0123478e75c6b09ce', 'Изменена ревизия')
require(MANIFEST['history_weeks'] == list(range(25)), 'История не 0–24')
require(CFG['sample_users'] == MANIFEST['sample_users_requested'] == 3000
        and CFG['sample_items'] == MANIFEST['sample_items_requested'] == 2000,
        'Сдвинуты размеры выборки')
require(CFG['random_seed'] == MANIFEST['seed_users'] == 42
        and MANIFEST['seed_items'] == 43, 'Изменены seeds формирования выборки')
require(MANIFEST['interactions_rows'] == 1454216, 'Изменён исходный train')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.fm import FactorizationMachine, prepare_fm_data, FEATURE_SETS
from models.fm_wns import SampledFactorizationMachine, UniformUnobservedSampler, weighted_batch_loss
from models.utils.evaluation import label_relevance, evaluate_scores, bootstrap_mean, bootstrap_paired_difference
require(tuple(FEATURE_SETS) == ('id','item_meta','user_meta','full'), 'Другой контракт feature sets')
print('PASS: данные и общий evaluator совпадают; WNS SHA-256:', module_hashes['models/fm_wns.py'])


TRAIN_COLS = ['user_id','item_id','week','timespent','like','bookmark','dislike']
VAL_COLS = ['user_id','item_id','timespent','like','bookmark','dislike']
train_raw = pd.read_parquet(FILES['train'], columns=TRAIN_COLS)
users_meta = pd.read_parquet(FILES['users'], columns=['user_id','age','gender','geo'])
items_meta = pd.read_parquet(FILES['items'], columns=['item_id','author_id','duration'])
val_raw = pd.read_parquet(FILES['validation'], columns=VAL_COLS)
for label, frame, keys in [('train',train_raw,['user_id','item_id']),
                           ('user metadata',users_meta,['user_id']),
                           ('item metadata',items_meta,['item_id']),
                           ('validation source',val_raw,['user_id','item_id'])]:
    require(not frame.empty and frame[keys].notna().all().all(), f'{label}: пусто или отсутствует ID')
    for k in keys:
        require(frame[k].dtype.kind in 'iu', f'{label}.{k}: ID преобразован в нецелое')
    require(not frame.duplicated(keys).any(), f'{label}: дубли по ID')
require(len(train_raw) == 1454216 and set(train_raw.week.unique()) == set(range(25)),
        'Train отличается от EDA')
require(len(users_meta)==3000 and len(items_meta)==2000 and len(val_raw)==519839,
        'Metadata / исходная validation не совпадают с аудитом')
USER_IDS = np.sort(users_meta.user_id.to_numpy(dtype=np.uint64))
ITEM_IDS = np.sort(items_meta.item_id.to_numpy(dtype=np.uint64))
user_hash, item_hash = sha256_uint64_ids(USER_IDS), sha256_uint64_ids(ITEM_IDS)
require(user_hash == EXPECTED_ID_SHA256['users'] and item_hash == EXPECTED_ID_SHA256['items'],
        'ID hash отличается от исходной EDA-выборки')
require(set(train_raw.user_id)==set(USER_IDS) and set(train_raw.item_id)==set(ITEM_IDS),
        'История train использует другой каталог')
validation_raw = val_raw.loc[val_raw.user_id.isin(USER_IDS) & val_raw.item_id.isin(ITEM_IDS)].copy()
validation_raw['week'] = np.int16(25)  # источник контролируется именем и SHA-256
require(len(validation_raw)==16158 and validation_raw.user_id.nunique()==2819
        and validation_raw.item_id.nunique()==839, 'Неверная validation фильтрация')
require(not validation_raw.duplicated(['user_id','item_id']).any(), 'Дубли в val')
require(not pd.MultiIndex.from_frame(validation_raw[['user_id','item_id']]).isin(
    pd.MultiIndex.from_frame(train_raw[['user_id','item_id']])).any(), 'Пересечение train/val')
seen = train_raw.groupby('user_id').item_id.nunique().reindex(USER_IDS,fill_value=0)
unseen = len(ITEM_IDS)-seen
require(int(unseen.min())==1124 and float(unseen.median())==1512 and (unseen>=10).all(),
        'Иной candidate pool')
print('PASS: users',len(USER_IDS),'items',len(ITEM_IDS),'train',len(train_raw),
      'val filtered',len(validation_raw),'positive candidate min',int(unseen.min()))


item_duration = items_meta[['item_id','duration']]
require(items_meta.duration.notna().all() and items_meta.duration.gt(0).all(),
        'Некорректные duration в item metadata')
train_joined = train_raw.merge(item_duration,on='item_id',how='left',validate='many_to_one',sort=False)
val_joined = validation_raw.merge(item_duration,on='item_id',how='left',validate='many_to_one',sort=False)
require(len(train_joined)==1454216 and len(val_joined)==16158
        and train_joined.duration.notna().all() and val_joined.duration.notna().all(),
        'Ошибка merge durations')
train_labeled = label_relevance(train_joined)
val_labeled = label_relevance(val_joined)
train_levels = train_labeled.rel.value_counts().reindex([0,1,2],fill_value=0)
val_levels = val_labeled.rel.value_counts().reindex([0,1,2],fill_value=0)
require(np.array_equal(train_levels.to_numpy(),[922110,148820,383286]), 'Train rel изменён')
require(np.array_equal(val_levels.to_numpy(),[11352,1462,3344]), 'Val rel изменён')
require(int((val_labeled.rel>=1).sum())==4806
        and val_labeled.loc[val_labeled.rel>=1,'user_id'].nunique()==2082,
        'Val eligible изменился')
HISTORY = train_labeled[['user_id','item_id','week','rel']].copy()
HOLDOUT = val_labeled[['user_id','item_id','week','rel']].copy()
require(HISTORY.week.between(0,24).all() and HOLDOUT.week.eq(25).all(), 'Неверные недели')
label_table = pd.DataFrame({'rel':[0,1,2],'train':train_levels.to_numpy(),
                            'validation':val_levels.to_numpy()})
display(label_table)
print('PASS: 532106 positive train; 4806 positive val, 2082 eligible')
del train_raw, val_raw, validation_raw, train_joined, val_joined, train_labeled, val_labeled
_ = gc.collect()


tiny_u = np.array([2**63+17,2**63+21,2**63+29],dtype=np.uint64)
tiny_i = np.arange(2**63+100,2**63+116,dtype=np.uint64)
tiny_um = pd.DataFrame({'user_id':tiny_u,'age':[24,35,43],
                        'gender':[0,1,1],'geo':[3,4,3]})
tiny_im = pd.DataFrame({'item_id':tiny_i,'author_id':[11,12]*8,
                        'duration':[15+2*j for j in range(16)]})
tiny_h = pd.DataFrame({'user_id':tiny_u[[0,0,1,1,2,2]],
                       'item_id':tiny_i[[0,1,1,2,2,3]],
                       'week':[0,1,2,3,3,4],'rel':[0,2,1,0,2,1]})
tiny_val = pd.DataFrame({'user_id':tiny_u,'item_id':tiny_i[[4,5,6]],
                         'week':[5,5,5],'rel':[2,1,2]})
tiny_prepared = prepare_fm_data(history=tiny_h,users_metadata=tiny_um,items_metadata=tiny_im,
    user_ids=tiny_u,item_ids=tiny_i,feature_set='full',history_start_week=0,history_end_week=4)
tiny_m = SampledFactorizationMachine(n_factors=4,batch_size=2,seed=42,device='cpu',
    negative_ratio=1.,negative_weight=.1,sampling_seed=1042)
tiny_m.fit_prepared(tiny_prepared,epochs=1)
tiny_scores = tiny_m.score(batch_size=19)
tiny_eval = evaluate_scores(tiny_scores,tiny_u,tiny_i,tiny_h,tiny_val,k=10,
                            history_start_week=0,history_end_week=4,holdout_week=5)
require(tiny_scores.shape == (3,16) and np.isfinite(tiny_scores).all(), 'FM score invalid')
require(len(tiny_eval.recommendations)==30 and len(tiny_eval.per_user)==3,
        'Evaluator integration invalid')
require(not pd.MultiIndex.from_frame(tiny_eval.recommendations[['user_id','item_id']]).isin(
    pd.MultiIndex.from_frame(tiny_h[['user_id','item_id']])).any(),
    'Известные items не исключены')
tiny_sampler = UniformUnobservedSampler(tiny_prepared)
tiny_unknown = tiny_sampler.sample(negative_ratio=1.,rng=np.random.default_rng(1042))
seen_flat = set((np.where(tiny_u==u)[0][0]*len(tiny_i)+np.where(tiny_i==i)[0][0])
                for u,i in zip(tiny_h.user_id,tiny_h.item_id))
require(len(tiny_unknown)==len(tiny_h) and len(set(tiny_unknown))==len(tiny_unknown)
        and not seen_flat.intersection(set(tiny_unknown)), 'Sampler включает известные пары')
with __import__('tempfile').TemporaryDirectory() as tmp:
    pt=Path(tmp)/'toy.pt'
    tiny_m.save_checkpoint(pt)
    restored=SampledFactorizationMachine.load_checkpoint(pt,prepared=tiny_prepared,
                       device='cpu',expected_config=tiny_m._config())
    require(np.array_equal(tiny_m.score(),restored.score()), 'Ошибка checkpoint WNS')
original = FactorizationMachine(n_factors=4,batch_size=2,seed=42,device='cpu')
original.fit_prepared(tiny_prepared,epochs=1)
control = SampledFactorizationMachine(n_factors=4,batch_size=2,seed=42,device='cpu',
    negative_weight=0.,negative_ratio=1.)
control.fit_prepared(tiny_prepared,epochs=1)
require(np.array_equal(original.score(),control.score()), 'β=0 не совпадает с исходной FM')
print('PASS: sampler + checkpoint + baseline parity + official evaluator, huge uint64')
del tiny_prepared,tiny_m,tiny_scores,tiny_eval,tiny_sampler,restored,control,original
_ = gc.collect()


t_prepare = time.perf_counter()
PREPARED = prepare_fm_data(history=HISTORY, users_metadata=users_meta,
    items_metadata=items_meta, user_ids=USER_IDS, item_ids=ITEM_IDS,
    feature_set='full', history_start_week=0, history_end_week=24)
PREPARE_SECONDS = time.perf_counter() - t_prepare
require(len(PREPARED.labels)==1454216 and int(PREPARED.labels.sum())==532106,
        'Изменился подготовленный train')
require(PREPARED.encoder.n_features>5000 and len(PREPARED.encoder.fields)==7,
        'Некорректные Full FM features')
N_OBSERVED = len(PREPARED.labels)
N_UNKNOWN = len(USER_IDS)*len(ITEM_IDS)-N_OBSERVED
N_SAMPLED = int(np.floor(NEGATIVE_RATIO*N_OBSERVED + 0.5))
require(N_SAMPLED<=N_UNKNOWN, 'Sampling ratio превышает доступное число неизвестных пар')
print('D=', PREPARED.encoder.n_features, '| N=',N_OBSERVED,
      '| unknown=',N_UNKNOWN,'| sampled/epoch=',N_SAMPLED,
      '| prepare seconds=',round(PREPARE_SECONDS,1))
print('При beta>0 количество AdamW steps/epoch:',
      (N_OBSERVED+N_SAMPLED+BATCH_SIZE-1)//BATCH_SIZE,
      '| при beta=0:',(N_OBSERVED+BATCH_SIZE-1)//BATCH_SIZE)

PREPARED_BY_FEATURE = {'full': PREPARED}


CACHE.mkdir(parents=True, exist_ok=True)
REVIEW.mkdir(parents=True, exist_ok=True)

CONFIG_BASE = dict(n_factors=N_FACTORS, learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, negative_ratio=NEGATIVE_RATIO,
    sampling_strategy='uniform_unobserved', max_catalog_pairs=10_000_000)


def make_config(beta, seed):
    return {**CONFIG_BASE, 'seed':int(seed), 'sampling_seed':int(seed)+1000,
            'negative_weight':float(beta)}


def get_prepared(feature_set):
    require(feature_set in FEATURE_SETS, 'Неизвестный feature set')
    if feature_set not in PREPARED_BY_FEATURE:
        t=time.perf_counter()
        x=prepare_fm_data(history=HISTORY,users_metadata=users_meta,
            items_metadata=items_meta,user_ids=USER_IDS,item_ids=ITEM_IDS,
            feature_set=feature_set,history_start_week=0,history_end_week=24)
        require(len(x.labels)==len(HISTORY) and int(x.labels.sum())==532106,
                'Некорректный обучающий target для '+feature_set)
        PREPARED_BY_FEATURE[feature_set]=x
        print('Prepared',feature_set,'D=',x.encoder.n_features,
              'time=',round(time.perf_counter()-t,1),'s')
    return PREPARED_BY_FEATURE[feature_set]

RUN_IDENTITY={
    'experiment':'VK-LSVD-FM-WNS-robustness-ablation-v1',
    'source_revision':REVISION['revision'], 'data_sha256':file_hashes,
    'module_sha256':module_hashes, 'user_sha256':user_hash, 'item_sha256':item_hash,
    'full_prepared_fingerprint':PREPARED.fingerprint,
    'beta':list(NEGATIVE_WEIGHTS), 'seeds':list(SEEDS),'epochs':list(EPOCHS),
    'ablation_feature_sets':list(ABLATION_FEATURE_SETS),
    'ablation_beta':ABLATION_BETA,'ablation_seed':ABLATION_SEED,
    'config_base':CONFIG_BASE,'device':EXPERIMENT_DEVICE, 'top_k':TOP_K,
    'torch':str(torch.__version__), 'numpy':np.__version__, 'pandas':pd.__version__,
    'python':platform.python_version(), 'train_window':[0,24], 'validation_week':25,
}
IDENTITY_HASH=hashlib.sha256(json.dumps(RUN_IDENTITY,sort_keys=True,
    separators=(',',':')).encode()).hexdigest()
identity_path=CACHE/'run_identity.json'
if identity_path.exists():
    require(json.loads(identity_path.read_text(encoding='utf-8'))==
        {'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY},
        'Кэш относится к другой версии данных/модели/параметров. Не смешивать.')
else:
    require(not list(CACHE.iterdir()),'Неизвестный кэш без run_identity.json')
    write_json_atomic(identity_path,{'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY})


def run_stem(feature_set,beta,seed):
    return f'wns_{feature_set}_b{float(beta):.4f}_s{int(seed)}_r{NEGATIVE_RATIO:.4f}'


def cp_files(feature_set,beta,seed,epoch):
    name=run_stem(feature_set,beta,seed)+f'_ep{int(epoch):02d}'
    base=CACHE/name
    return {'pt':Path(str(base)+'.pt'),
            'per_user':Path(str(base)+'_per_user.parquet'),
            'recommendations':Path(str(base)+'_recommendations.parquet'),
            'marker':Path(str(base)+'_complete.json')}


def as_metrics(ev):
    s=ev.summary
    return {'n_users':s['n_users'],'n_eligible':s['n_eligible'],
            'eligible_ndcg_at_10':s['eligible']['ndcg_at_10'],
            'eligible_recall_at_10':s['eligible']['recall_at_10'],
            'eligible_map_at_10':s['eligible']['map_at_10'],
            'all_users_ndcg_at_10':s['all_users']['ndcg_at_10'],
            'all_users_recall_at_10':s['all_users']['recall_at_10'],
            'all_users_map_at_10':s['all_users']['map_at_10']}


def rank_diagnostics(recs):
    counts=recs.groupby('item_id',sort=False).size().sort_values(ascending=False)
    return {'unique_recommended_items':int(len(counts)),
            'top100_slot_share':float(counts.head(100).sum()/len(recs)),
            'top10_slot_share':float(counts.head(10).sum()/len(recs))}


def completed_record(feature_set,beta,seed,epoch,prepared):
    files=cp_files(feature_set,beta,seed,epoch)
    if not files['marker'].is_file():
        return None
    rec=json.loads(files['marker'].read_text(encoding='utf-8'))
    require(rec.get('identity_sha256')==IDENTITY_HASH and
            rec.get('prepared_fingerprint')==prepared.fingerprint and
            rec.get('config')==make_config(beta,seed) and
            rec.get('feature_set')==feature_set and rec.get('epoch')==int(epoch),
            'Кэш checkpoint относится к другому обучению: '+str(files['marker']))
    for suffix in ('pt','per_user','recommendations'):
        require(files[suffix].is_file() and sha256_file(files[suffix])==rec['sha256'][suffix],
                'Ошибка кэша: '+str(files[suffix]))
    return rec


def progress(row):
    print('    epoch',row['epoch'], 'observed BCE=',round(row['train_bce'],5),
          'objective=',round(row['train_objective'],5),
          'time=',round(row['seconds'],2),'s',flush=True)


def train_and_evaluate(feature_set,beta,seed):
    prepared=get_prepared(feature_set)
    cfg=make_config(beta,seed)
    model=None
    records=[]
    for epoch in EPOCHS:
        f=cp_files(feature_set,beta,seed,epoch)
        saved=completed_record(feature_set,beta,seed,epoch,prepared)
        if saved is not None:
            records.append(saved)
            model=None
            print('  CACHED',run_stem(feature_set,beta,seed),'ep',epoch,flush=True)
            continue
        if f['pt'].exists():
            model=SampledFactorizationMachine.load_checkpoint(f['pt'],prepared=prepared,
                  device=EXPERIMENT_DEVICE,expected_config=cfg)
            require(model.trained_epochs==epoch,'Partial .pt не той эпохи')
        else:
            if model is None:
                checkpoints=[(r['epoch'],cp_files(feature_set,beta,seed,r['epoch'])['pt'])
                             for r in records if r['epoch']<epoch]
                if checkpoints:
                    model=SampledFactorizationMachine.load_checkpoint(
                        max(checkpoints,key=lambda x:x[0])[1],prepared=prepared,
                        device=EXPERIMENT_DEVICE,expected_config=cfg)
                else:
                    model=SampledFactorizationMachine(**cfg,device=EXPERIMENT_DEVICE,
                                                       progress_callback=progress)
                    model.fit_prepared(prepared,epochs=epoch)
            model.progress_callback=progress
            if model.trained_epochs < epoch:
                model.continue_fit(epoch-model.trained_epochs)
            require(model.trained_epochs==epoch,'Модель обучена не до нужной эпохи')
            model.save_checkpoint(f['pt'])
        require(model.trained_epochs==epoch,'Неверная эпоха checkpoint')
        scores=model.score(batch_size=BATCH_SIZE)
        require(scores.shape==(len(USER_IDS),len(ITEM_IDS)) and np.isfinite(scores).all(),
                'Score содержит NaN/Inf или неверную форму')
        scores_seconds=float(model.last_score_seconds)
        start=time.perf_counter()
        ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
            history=HISTORY,holdout=HOLDOUT,k=TOP_K,
            history_start_week=0,history_end_week=24,holdout_week=25)
        eval_seconds=time.perf_counter()-start
        require(len(ev.per_user)==len(USER_IDS) and len(ev.recommendations)==TOP_K*len(USER_IDS)
                and int(ev.per_user.eligible.sum())==2082,
                'Evaluator вернул несоответствующее число пользователей')
        require(np.array_equal(ev.per_user.user_id.to_numpy(dtype=np.uint64),USER_IDS),
                'Нарушился порядок user IDs')
        ev.per_user.to_parquet(f['per_user'],index=False)
        ev.recommendations.to_parquet(f['recommendations'],index=False)
        logs=[dict(row) for row in model.training_log]
        row={
            'identity_sha256':IDENTITY_HASH,'prepared_fingerprint':prepared.fingerprint,
            'config':cfg,'feature_set':feature_set,'epoch':int(epoch),
            **as_metrics(ev),**rank_diagnostics(ev.recommendations),
            'n_users_with_positive_hit':int((ev.per_user['recall_at_10']>0).sum()),
            'n_positive_hits':int(np.rint((ev.per_user['recall_at_10']*ev.per_user['n_relevant']).sum())),
            'train_bce_at_epoch':float(logs[-1]['train_bce']),
            'train_objective_at_epoch':float(logs[-1]['train_objective']),
            'sampled_bce_at_epoch':float(logs[-1]['sampled_bce_preupdate']),
            'optimizer_steps_total':int(sum(r['n_optimizer_steps'] for r in logs)),
            'n_observed':int(logs[-1]['n_observed']),
            'n_sampled':int(logs[-1]['n_sampled']),
            'fit_seconds_cumulative':float(sum(r['seconds'] for r in logs)),
            'scoring_seconds':scores_seconds,'evaluator_seconds':eval_seconds,
            'training_log':logs,
            'sha256':{k:sha256_file(f[k]) for k in ('pt','per_user','recommendations')},
        }
        write_json_atomic(f['marker'],row)
        print('  PASS',feature_set,'β',beta,'seed',seed,'epoch',epoch,
              'NDCG',round(row['eligible_ndcg_at_10'],6),flush=True)
        records.append(row)
        del scores,ev
        gc.collect()
    del model
    gc.collect()
    return records


def row_from_record(rec):
    cfg=rec['config']
    fields=('eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10',
            'all_users_ndcg_at_10','all_users_recall_at_10','all_users_map_at_10',
            'train_bce_at_epoch','train_objective_at_epoch','sampled_bce_at_epoch',
            'optimizer_steps_total','n_observed','n_sampled','n_users','n_eligible',
            'unique_recommended_items','top100_slot_share','top10_slot_share',
            'n_users_with_positive_hit','n_positive_hits',
            'fit_seconds_cumulative','scoring_seconds','evaluator_seconds')
    return {'feature_set':rec['feature_set'],'negative_weight':cfg['negative_weight'],
            'negative_ratio':cfg['negative_ratio'],'seed':cfg['seed'],
            'sampling_seed':cfg['sampling_seed'],'epoch':rec['epoch'],
            **{k:rec[k] for k in fields}}

print('PASS: checkpoint/caching identity',IDENTITY_HASH)


ROBUST_RECORDS=[]
if RUN_TRAINING:
    for beta in NEGATIVE_WEIGHTS:
        for seed in SEEDS:
            print('\n=== Robustness: full / beta',beta,'/ seed',seed,'===',flush=True)
            ROBUST_RECORDS.extend(train_and_evaluate('full',beta,seed))
require(len(ROBUST_RECORDS)==len(NEGATIVE_WEIGHTS)*len(SEEDS)*len(EPOCHS),
        'Неполная таблица seed robustness')
ROBUST=pd.DataFrame([row_from_record(x) for x in ROBUST_RECORDS]).sort_values(
    ['negative_weight','seed','epoch']).reset_index(drop=True)
require(not ROBUST.duplicated(['negative_weight','seed','epoch']).any(),
        'Дубликаты траекторий seed')
ROBUST_SUMMARY=ROBUST.groupby(['negative_weight','epoch'],as_index=False).agg(
    mean_ndcg=('eligible_ndcg_at_10','mean'),std_ndcg=('eligible_ndcg_at_10','std'),
    min_ndcg=('eligible_ndcg_at_10','min'),max_ndcg=('eligible_ndcg_at_10','max'),
    mean_recall=('eligible_recall_at_10','mean'),mean_map=('eligible_map_at_10','mean'),
    mean_coverage=('unique_recommended_items','mean'),
    mean_top100_share=('top100_slot_share','mean'))
require(len(ROBUST_SUMMARY)==len(NEGATIVE_WEIGHTS)*len(EPOCHS),
        'Неполная агрегированная таблица')
print('Seed robustness: NDCG среднего трёх seeds (и std между seeds):')
display(ROBUST_SUMMARY)


ABLATION_RECORDS=[]
if RUN_TRAINING:
    for feature_set in ABLATION_FEATURE_SETS:
        if feature_set=='full':
            selected=[r for r in ROBUST_RECORDS if r['feature_set']=='full'
                      and r['config']['negative_weight']==ABLATION_BETA
                      and r['config']['seed']==ABLATION_SEED]
            require(len(selected)==len(EPOCHS),'Не найдена Full β=5 seed42 для ablation')
            ABLATION_RECORDS.extend(selected)
            continue
        print('\n=== Ablation:',feature_set,'β=5 seed=42 ===',flush=True)
        ABLATION_RECORDS.extend(train_and_evaluate(feature_set,ABLATION_BETA,ABLATION_SEED))
require(len(ABLATION_RECORDS)==len(ABLATION_FEATURE_SETS)*len(EPOCHS),
        'Неполная ablation')
ABLATION=pd.DataFrame([row_from_record(r) for r in ABLATION_RECORDS]).sort_values(
    ['feature_set','epoch']).reset_index(drop=True)
require(not ABLATION.duplicated(['feature_set','epoch']).any(),
        'Дубликаты ablation')
print('Ablation при β=5, seed=42:')
display(ABLATION[['feature_set','epoch','eligible_ndcg_at_10',
                  'eligible_recall_at_10','eligible_map_at_10',
                  'unique_recommended_items','top100_slot_share']])


FIGURES={}
def safe_show(fig):
    try:
        fig.show()
    except (ValueError, ImportError) as exc:
        print('Inline Plotly недоступен:',exc,'; HTML будет сохранён в review_bundle')

for metric in ('eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10'):
    fig=px.line(ROBUST,x='negative_weight',y=metric,
                color=ROBUST.seed.astype(str),facet_col='epoch',markers=True,
                title='Seed robustness — '+metric,
                labels={'negative_weight':'Вес β','color':'Seed'})
    fig.update_layout(height=480)
    safe_show(fig)
    FIGURES['robust_'+metric]=fig

fig=px.line(ROBUST_SUMMARY,x='negative_weight',y='mean_ndcg',color=ROBUST_SUMMARY.epoch.astype(str),
     markers=True,title='Средний NDCG@10 по seeds 42/43/44',
     labels={'negative_weight':'β','mean_ndcg':'Mean Eligible NDCG@10','color':'Эпоха'})
safe_show(fig)
FIGURES['robust_mean_ndcg']=fig

fig=px.bar(ABLATION,x='feature_set',y='eligible_ndcg_at_10',color=ABLATION.epoch.astype(str),
    barmode='group',title='Ablation FM WNS β=5: вклад признаков',
    labels={'feature_set':'Набор признаков','eligible_ndcg_at_10':'NDCG@10','color':'Эпоха'})
safe_show(fig)
FIGURES['ablation_ndcg']=fig

fig=px.scatter(ROBUST,x='top100_slot_share',y='eligible_ndcg_at_10',
    color=ROBUST.negative_weight.astype(str),symbol=ROBUST.seed.astype(str),
    hover_data=['epoch','unique_recommended_items'],
    title='Seed robustness: концентрация выдачи и NDCG')
safe_show(fig)
FIGURES['concentration']=fig

fig=px.scatter(ROBUST,x='train_bce_at_epoch',y='eligible_ndcg_at_10',
    color=ROBUST.negative_weight.astype(str),symbol=ROBUST.seed.astype(str),
    hover_data=['epoch'],title='Наблюдавшаяся BCE против ranking quality')
safe_show(fig)
FIGURES['observed_bce_vs_ndcg']=fig


def read_per_user(feature_set,beta,seed,epoch):
    f=cp_files(feature_set,beta,seed,epoch)['per_user']
    require(f.is_file(), 'Отсутствует per-user: '+str(f))
    df=pd.read_parquet(f)
    require(len(df)==len(USER_IDS) and not df.user_id.duplicated().any()
            and np.array_equal(df.user_id.to_numpy(dtype=np.uint64),USER_IDS)
            and int(df.eligible.sum())==2082,'Per-user нарушает единый протокол')
    return df

PAIRED_ROWS=[]
if RUN_PAIRWISE_BOOTSTRAP:
    for seed in SEEDS:
        reference=read_per_user('full',5.,seed,25)
        for beta in (1.,10.,20.):
            delta=bootstrap_paired_difference(read_per_user('full',beta,seed,25),
                reference,metric='ndcg_at_10',group='eligible',
                n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            PAIRED_ROWS.append({'comparison':f'full β={beta:g} − β=5 / seed={seed} / ep25',**delta})
    ablation_full=read_per_user('full',5.,42,25)
    for feature_set in ('id','item_meta','user_meta'):
        delta=bootstrap_paired_difference(ablation_full,
            read_per_user(feature_set,5.,42,25),metric='ndcg_at_10',group='eligible',
            n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        PAIRED_ROWS.append({'comparison':f'full − {feature_set} / β=5 seed42 ep25',**delta})
PAIRED=pd.DataFrame(PAIRED_ROWS)
if not PAIRED.empty:
    display(PAIRED[['comparison','estimate','ci_low','ci_high']])

# Предыдущий β=5, seed=42, epoch=25: только при совпадении конфигурации и источников.
PRIOR_BETA_STATUS='Исторический β=5 не найден или не проверен; эксперимент автономен.'
old_dir=ROOT/'results/fm/wns_beta_above_one_10_to_25/review_bundle'
old_m=old_dir/'selection_and_manifest.json'
old_csv=old_dir/'all_18_validation_metrics.csv'
if old_m.is_file() and old_csv.is_file():
    old=json.loads(old_m.read_text(encoding='utf-8'))
    compatible=(old.get('data_sha256')==file_hashes and
       old.get('module_sha256')==module_hashes and
       old.get('user_id_sha256')==user_hash and old.get('item_id_sha256')==item_hash and
       old.get('device')==EXPERIMENT_DEVICE and old.get('n_factors')==N_FACTORS and
       old.get('learning_rate')==LEARNING_RATE and old.get('weight_decay')==WEIGHT_DECAY and
       old.get('batch_size')==BATCH_SIZE and old.get('negative_ratio')==NEGATIVE_RATIO and
       old.get('seed')==42 and old.get('sampling_seed')==1042)
    if compatible:
        old_rows=pd.read_csv(old_csv)
        old_match=old_rows.loc[np.isclose(old_rows.negative_weight,5.) & old_rows.epoch.eq(25)]
        require(len(old_match)==1,'Не найден единственный исторический β=5 ep25')
        cur=ROBUST.loc[(ROBUST.negative_weight==5.)&(ROBUST.seed==42)&(ROBUST.epoch==25)].iloc[0]
        prior=float(old_match.iloc[0].eligible_ndcg_at_10)
        current=float(cur.eligible_ndcg_at_10)
        PRIOR_BETA_STATUS=f'Исторический β=5: NDCG={prior:.8f}; повторный β=5 seed42={current:.8f}'
        if np.isclose(prior,current,rtol=0,atol=1e-9):
            PRIOR_BETA_STATUS+='; результат воспроизведён.'
        else:
            PRIOR_BETA_STATUS+='; результат отличается: проверить версии и детерминизм MPS.'
    else:
        PRIOR_BETA_STATUS='Исторический β=5 есть, но fingerprint/устройство отличаются: сравнение не проводим.'
print(PRIOR_BETA_STATUS)


REVIEW.mkdir(parents=True,exist_ok=True)
ROBUST.to_csv(REVIEW/'seed_robustness_all_24.csv',index=False)
ROBUST_SUMMARY.to_csv(REVIEW/'seed_robustness_mean_std.csv',index=False)
ABLATION.to_csv(REVIEW/'ablation_all_8.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_differences.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)

all_new={}
for r in [*ROBUST_RECORDS,*ABLATION_RECORDS]:
    cfg=r['config']
    key=(r['feature_set'],float(cfg['negative_weight']),int(cfg['seed']),int(r['epoch']))
    all_new[key]=r
require(len(all_new)==30,'Должно быть 30 уникальных оценок, Full ablation не должна дублироваться')
ALL_RESULTS=pd.DataFrame([row_from_record(x) for x in all_new.values()]).sort_values(
    ['feature_set','negative_weight','seed','epoch']).reset_index(drop=True)
ALL_RESULTS.to_csv(REVIEW/'all_30_validation_metrics.csv',index=False)

logs=[]
personal=[]
for (feature,beta,seed,ep),r in all_new.items():
    df=read_per_user(feature,beta,seed,ep).copy()
    df.insert(0,'feature_set',feature)
    df.insert(1,'negative_weight',beta)
    df.insert(2,'seed',seed)
    df.insert(3,'epoch',ep)
    personal.append(df)
    for log in r['training_log']:
        logs.append({'feature_set':feature,'negative_weight':beta,'seed':seed,**log})
ALL_PERSONAL=pd.concat(personal,ignore_index=True)
require(len(ALL_PERSONAL)==len(USER_IDS)*30,'Неверное число per-user записей')
ALL_PERSONAL.to_parquet(REVIEW/'all_30_per_user.parquet',index=False,compression='zstd')
TRAINING_LOGS=(pd.DataFrame(logs).drop_duplicates(['feature_set','negative_weight','seed','epoch'])
    .sort_values(['feature_set','negative_weight','seed','epoch']).reset_index(drop=True))
require(len(TRAINING_LOGS)==15*25,'Неверное число epoch logs: ожидается 375')
TRAINING_LOGS.to_csv(REVIEW/'all_training_epochs.csv',index=False)

# Выбор β только по среднему по seeds при конкретной эпохе, ablation фиксирована β=5.
selected=ROBUST_SUMMARY.sort_values(['mean_ndcg','negative_weight','epoch'],
    ascending=[False,True,True]).iloc[0]
selected_beta=float(selected.negative_weight)
selected_epoch=int(selected.epoch)
SELECTED_PU=read_per_user('full',selected_beta,42,selected_epoch)
SELECTED_PU.to_csv(REVIEW/'selected_full_seed42_per_user.csv',index=False)
recs=pd.read_parquet(cp_files('full',selected_beta,42,selected_epoch)['recommendations'])
require(len(recs)==TOP_K*len(USER_IDS),'Неполная выдача выбранной конфигурации')
recs.to_csv(REVIEW/'selected_full_seed42_recommendations.csv',index=False)

for key,fig in FIGURES.items():
    fig.write_html(str(REVIEW/f'wns_{key}.html'),include_plotlyjs='cdn',full_html=True)

manifest_out={
    'status':'PASS_VALIDATION_ONLY','experiment':'VK-LSVD FM WNS robustness + ablation v1.0',
    'identity_sha256':IDENTITY_HASH,'data_sha256':file_hashes,
    'module_sha256':module_hashes,'user_id_sha256':user_hash,'item_id_sha256':item_hash,
    'source_revision':REVISION['revision'],'device':EXPERIMENT_DEVICE,
    'n_users':len(USER_IDS),'n_items':len(ITEM_IDS),
    'train_rows':len(HISTORY),'val_rows':len(HOLDOUT),'val_eligible':2082,
    'full_feature_set':'full','n_factors':N_FACTORS,'learning_rate':LEARNING_RATE,
    'weight_decay':WEIGHT_DECAY,'batch_size':BATCH_SIZE,
    'negative_ratio':NEGATIVE_RATIO,'negative_weights':list(NEGATIVE_WEIGHTS),
    'seeds':list(SEEDS),'sampling_seed_rule':'seed+1000',
    'epochs':list(EPOCHS),'ablation_beta':ABLATION_BETA,'ablation_seed':ABLATION_SEED,
    'ablation_features':list(ABLATION_FEATURE_SETS),
    'training_trajectories':15,'validation_evaluations':len(ALL_RESULTS),
    'selection_rule':'maximum mean eligible NDCG across 3 seeds per (beta,epoch)',
    'selected_mean_config':{'negative_weight':selected_beta,'epoch':selected_epoch,
                             'mean_ndcg':float(selected.mean_ndcg),
                             'std_ndcg':float(selected.std_ndcg)},
    'historical_status':PRIOR_BETA_STATUS,
    'bootstrap_n':BOOTSTRAP_N,'bootstrap_seed':BOOTSTRAP_SEED,
    'bootstrap_post_selection_caveat':True,
    'notes':'Each beta/seed/feature set from scratch; no future data, no test; ablation beta=5 fixed before search.',
    'train_weeks':'0-24','validation_week':25,'test_read':False,
    'test_evaluated':False,'final_refit':False,
}
write_json_atomic(REVIEW/'selection_and_manifest.json',manifest_out)
(REVIEW/'README_results.md').write_text(
    '# VK-LSVD · FM WNS β 1,5,10,20 · Seed robustness и metadata ablation\n\n'
    'Seed robustness: 4 weights x 3 seeds (42,43,44) x epochs (15,25) = 24 evaluations. '
    'Ablation: fixed beta=5, seed=42, four feature sets x 2 epochs = 8 evaluations '
    '(Full reused). Total: 15 independent fits, 30 unique validations, 375 training epochs. '
    'All train weeks 0-24, validation week25; test never opened. Model/evaluator unchanged. '
    'Selection by mean NDCG across seeds, not a new independent validation. '
    'Bootstrap after selection is descriptive. Checkpoints remain locally in checkpoint_cache.\n',
    encoding='utf-8')
require((REVIEW/'all_30_per_user.parquet').is_file() and
    len(ALL_RESULTS)==30 and len(TRAINING_LOGS)==375,'Неполный review bundle')
print('PASS. Готов:', REVIEW)
print('Результаты сохранены.')
print('Test не открывался. Результаты получены ТОЛЬКО на validation.')
