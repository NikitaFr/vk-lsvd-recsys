"""fm wns beta large experiment on train/validation."""
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
OUT = ROOT/'results/fm/wns_beta_above_one_10_to_25'
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
NEGATIVE_WEIGHTS = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0)
NEGATIVE_RATIO = 1.0
EPOCHS = (10, 15, 25)
TOP_K = 10
N_FACTORS = 64
LEARNING_RATE = .003
WEIGHT_DECAY = .1
BATCH_SIZE = 8192
SEED = 42
SAMPLING_SEED = 1042
BOOTSTRAP_N, BOOTSTRAP_SEED = 2000, 42
EXPERIMENT_DEVICE = 'mps' if FM_MPS_VALIDATED else 'cpu'
# По необходимости можно выбрать 'cpu' вручную до запуска тренировок.
RUN_TRAINING = True
RUN_PAIRWISE_BOOTSTRAP = True
REQUIRE_BASELINE_PARITY = False  # исторические данные доступны лишь для сравнения
print('Project:', ROOT)
print('PyTorch:', torch.__version__, '| device:', EXPERIMENT_DEVICE)
print('Weights:', NEGATIVE_WEIGHTS, '| epochs:', EPOCHS, '| ratio:', NEGATIVE_RATIO)


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


CACHE.mkdir(parents=True,exist_ok=True)
REVIEW.mkdir(parents=True,exist_ok=True)
BASE_CONFIG = dict(n_factors=N_FACTORS, learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, seed=SEED,
    negative_ratio=NEGATIVE_RATIO, sampling_seed=SAMPLING_SEED,
    sampling_strategy='uniform_unobserved', max_catalog_pairs=10_000_000)

def wns_config(beta):
    return {**BASE_CONFIG, 'negative_weight':float(beta)}

RUN_IDENTITY = {
    'experiment':'VK-LSVD-FM-WNS-beta-above-one-v1',
    'revision':REVISION['revision'], 'source_file_sha256':file_hashes,
    'module_sha256':module_hashes, 'user_id_sha256':user_hash,
    'item_id_sha256':item_hash,'prepared_fingerprint':PREPARED.fingerprint,
    'feature_set':'full', 'train_window':[0,24],'validation_week':25,
    'train_rows':len(HISTORY),'validation_rows':len(HOLDOUT),
    'negative_weights':list(NEGATIVE_WEIGHTS),'negative_ratio':NEGATIVE_RATIO,
    'epochs':list(EPOCHS),'base_config':BASE_CONFIG,'device':EXPERIMENT_DEVICE,
    'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,
    'python':platform.python_version(), 'top_k':TOP_K,
}
IDENTITY_HASH=hashlib.sha256(json.dumps(RUN_IDENTITY,sort_keys=True,
    separators=(',',':')).encode()).hexdigest()
IDENTITY_PATH=CACHE/'run_identity.json'
if IDENTITY_PATH.exists():
    require(json.loads(IDENTITY_PATH.read_text(encoding='utf-8')) ==
            {'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY},
            'Кэш создан с ДРУГИМИ данными/настройками/модулями! Не смешивать эксперименты.')
else:
    require(not list(CACHE.iterdir()), 'Кэш не пуст, но нет run_identity.json: данные ненадёжны.')
    write_json_atomic(IDENTITY_PATH,{'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY})

def run_stem(beta):
    return f'wns_beta_{float(beta):.4f}_ratio_{NEGATIVE_RATIO:.4f}_s{SEED}'

def cp_files(beta,epoch):
    base=CACHE/f'{run_stem(beta)}_ep{int(epoch):02d}'
    return {'pt':Path(str(base)+'.pt'),
            'per_user':Path(str(base)+'_per_user.parquet'),
            'recommendations':Path(str(base)+'_recommendations.parquet'),
            'marker':Path(str(base)+'_complete.json')}

def as_metrics(ev):
    return {'n_users':ev.summary['n_users'], 'n_eligible':ev.summary['n_eligible'],
            'eligible_ndcg_at_10':ev.summary['eligible']['ndcg_at_10'],
            'eligible_recall_at_10':ev.summary['eligible']['recall_at_10'],
            'eligible_map_at_10':ev.summary['eligible']['map_at_10'],
            'all_users_ndcg_at_10':ev.summary['all_users']['ndcg_at_10'],
            'all_users_recall_at_10':ev.summary['all_users']['recall_at_10'],
            'all_users_map_at_10':ev.summary['all_users']['map_at_10']}

def rank_diagnostics(recs):
    counts=recs.groupby('item_id',sort=False).size().sort_values(ascending=False)
    return {'unique_recommended_items':int(len(counts)),
            'top100_slot_share':float(counts.head(100).sum()/len(recs)),
            'top10_slot_share':float(counts.head(10).sum()/len(recs))}

def completed_record(beta,epoch):
    files=cp_files(beta,epoch)
    if not files['marker'].is_file():
        return None
    record=json.loads(files['marker'].read_text(encoding='utf-8'))
    require(record['identity_sha256']==IDENTITY_HASH
            and record['prepared_fingerprint']==PREPARED.fingerprint
            and record['config']==wns_config(beta) and record['epoch']==int(epoch),
            'Кэш checkpoint относится к другому запуску')
    for name in ('pt','per_user','recommendations'):
        require(files[name].is_file() and sha256_file(files[name])==record['sha256'][name],
                f'Повреждён кэш: {files[name]}')
    return record

def progress(row):
    print('   epoch',row['epoch'], 'observed BCE=',f"{row['train_bce']:.5f}",
          'weighted objective=',f"{row['train_objective']:.5f}",
          'time=',f"{row['seconds']:.1f}s",flush=True)

def train_and_evaluate(beta):
    records=[]
    cfg=wns_config(beta)
    model=None
    for epoch in EPOCHS:
        files=cp_files(beta,epoch)
        saved=completed_record(beta,epoch)
        if saved is not None:
            records.append(saved)
            model=None
            print('  CACHED',run_stem(beta),'ep',epoch,flush=True)
            continue
        # Partial checkpoint: .pt exists but marker/per-user incomplete → re-evaluate exactly the saved model.
        if files['pt'].is_file():
            model=SampledFactorizationMachine.load_checkpoint(files['pt'],
                    prepared=PREPARED,device=EXPERIMENT_DEVICE,expected_config=cfg)
        else:
            if model is None:
                previous = [(r['epoch'],cp_files(beta,r['epoch'])['pt'])
                            for r in records if r['epoch']<epoch]
                if previous:
                    previous_pt=max(previous,key=lambda x:x[0])[1]
                    model=SampledFactorizationMachine.load_checkpoint(previous_pt,
                        prepared=PREPARED,device=EXPERIMENT_DEVICE,expected_config=cfg)
                else:
                    model=SampledFactorizationMachine(**cfg,device=EXPERIMENT_DEVICE,
                                                      progress_callback=progress)
                    model.fit_prepared(PREPARED,epochs=epoch)
            model.progress_callback=progress
            if model.trained_epochs < epoch:
                model.continue_fit(epoch-model.trained_epochs)
            require(model.trained_epochs==epoch, 'Неверное число завершённых эпох')
            model.save_checkpoint(files['pt'])
        require(model.trained_epochs==epoch, 'Checkpoint содержит другую эпоху')
        print('  VALIDATION',run_stem(beta),'ep',epoch,flush=True)
        scores=model.score(batch_size=BATCH_SIZE)
        require(scores.shape==(len(USER_IDS),len(ITEM_IDS)) and np.isfinite(scores).all(),
                'FM вернула NaN/Inf или неверную форму scores')
        score_seconds=float(model.last_score_seconds)
        t0=time.perf_counter()
        ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
             history=HISTORY,holdout=HOLDOUT,k=TOP_K,
             history_start_week=0,history_end_week=24,holdout_week=25)
        eval_seconds=time.perf_counter()-t0
        require(len(ev.per_user)==len(USER_IDS) and int(ev.per_user.eligible.sum())==2082
                and len(ev.recommendations)==TOP_K*len(USER_IDS),
                'Единый evaluator вернул неожиданные размеры')
        require(np.array_equal(ev.per_user.user_id.to_numpy(dtype=np.uint64),USER_IDS),
                'Нарушен порядок пользователей')
        ev.per_user.to_parquet(files['per_user'],index=False)
        ev.recommendations.to_parquet(files['recommendations'],index=False)
        log=[dict(row) for row in model.training_log]
        rec={'identity_sha256':IDENTITY_HASH,'prepared_fingerprint':PREPARED.fingerprint,
             'config':cfg,'epoch':epoch,**as_metrics(ev),**rank_diagnostics(ev.recommendations),
             'n_users_with_positive_hit':int((ev.per_user['recall_at_10']>0).sum()),
             'n_positive_hits':int(np.rint((ev.per_user['recall_at_10']*ev.per_user['n_relevant']).sum())),
             'train_bce_at_epoch':float(log[-1]['train_bce']),
             'train_objective_at_epoch':float(log[-1]['train_objective']),
             'sampled_bce_at_epoch':None if beta==0 else float(log[-1]['sampled_bce_preupdate']),
             'optimizer_steps_total':int(sum(row['n_optimizer_steps'] for row in log)),
             'n_observed':int(log[-1]['n_observed']), 'n_sampled':int(log[-1]['n_sampled']),
             'fit_seconds_cumulative':float(sum(row['seconds'] for row in log)),
             'scoring_seconds':score_seconds,'evaluator_seconds':eval_seconds,
             'training_log':log,
             'sha256':{name:sha256_file(files[name])
                      for name in ('pt','per_user','recommendations')}}
        write_json_atomic(files['marker'],rec) # commit marker is LAST
        print('  PASS beta',beta,'epoch',epoch,'NDCG=',f"{rec['eligible_ndcg_at_10']:.6f}",flush=True)
        records.append(rec)
        del ev,scores
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
    return {'negative_weight':cfg['negative_weight'],
            'negative_ratio':cfg['negative_ratio'],'epoch':rec['epoch'],
            **{k:rec[k] for k in fields}}



# Исторические результаты — READ-ONLY, только для сравнения после обучения.
PREVIOUS_REVIEW = ROOT/'results/fm/wns_beta_extension_15_to_25/review_bundle'
PREVIOUS_CACHE = ROOT/'results/fm/wns_beta_extension_15_to_25/checkpoint_cache'
print('PASS WNS β>1 cache identity:', IDENTITY_HASH)
print('Все шесть β будут обучаться с нуля (или восстановлены из собственного нового кэша).')


RECORDS=[]
if RUN_TRAINING:
    for index,beta in enumerate(NEGATIVE_WEIGHTS,1):
        print(f'\n===== WNS {index}/{len(NEGATIVE_WEIGHTS)}: beta={beta:g}, ratio={NEGATIVE_RATIO:g} =====',flush=True)
        RECORDS.extend(train_and_evaluate(beta))
require(len(RECORDS)==len(NEGATIVE_WEIGHTS)*len(EPOCHS), 'Не все 18 точек получены')
RESULTS=pd.DataFrame([row_from_record(r) for r in RECORDS]).sort_values(
    ['negative_weight','epoch']).reset_index(drop=True)
require(not RESULTS.duplicated(['negative_weight','epoch']).any(), 'Дубли результатов')
selection=RESULTS.sort_values(['eligible_ndcg_at_10','negative_weight','epoch'],
                              ascending=[False,True,True]).iloc[0]
BEST_BETA=float(selection.negative_weight)
BEST_EPOCH=int(selection.epoch)
print('Лучшая проверенная конфигурация: beta=',BEST_BETA,'epoch=',BEST_EPOCH,
      'NDCG=',float(selection.eligible_ndcg_at_10))
display(RESULTS[['negative_weight','epoch','eligible_ndcg_at_10','eligible_recall_at_10',
                 'eligible_map_at_10','train_bce_at_epoch','train_objective_at_epoch',
                 'optimizer_steps_total','unique_recommended_items','top100_slot_share']])


FIGURES={}
PLOT_RESULTS=RESULTS.copy()
PLOT_RESULTS['beta_label']=PLOT_RESULTS['negative_weight'].map(lambda x:f'{x:g}')
def safe_show(fig):
    try:
        fig.show()
    except (ValueError, ImportError) as exc:
        print('Plotly inline недоступен:',exc,'(HTML сохраняется в review_bundle).')

for metric in ('eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10'):
    fig=px.line(PLOT_RESULTS,x='epoch',y=metric,color='beta_label',markers=True,
        title=f'WNS β=1.0–5.0: {metric} по эпохам',
        labels={'beta_label':'β','epoch':'Эпоха',metric:metric})
    fig.update_layout(height=480)
    safe_show(fig)
    FIGURES[metric]=fig

fig=px.line(PLOT_RESULTS, x='negative_weight',y='eligible_ndcg_at_10',
            color=PLOT_RESULTS.epoch.astype(str),markers=True,
            title='Влияние β>1 на NDCG@10 (10/15/25 эпох)',
            labels={'negative_weight':'Вес неизвестных пар β','color':'Эпоха'})
fig.update_xaxes(type='linear')
safe_show(fig)
FIGURES['beta_vs_ndcg']=fig

fig=px.scatter(PLOT_RESULTS,x='optimizer_steps_total',y='eligible_ndcg_at_10',
    color='beta_label',symbol='epoch',
    title='NDCG@10 относительно числа AdamW steps')
safe_show(fig)
FIGURES['optimizer_steps_vs_ndcg']=fig
fig=px.scatter(PLOT_RESULTS,x='top100_slot_share',y='eligible_ndcg_at_10',
    color='beta_label',symbol='epoch',hover_data=['unique_recommended_items'],
    title='Концентрация рекомендаций и NDCG@10')
safe_show(fig)
FIGURES['concentration_vs_ndcg']=fig
fig=px.scatter(PLOT_RESULTS,x='train_bce_at_epoch',y='eligible_ndcg_at_10',
    color='beta_label',symbol='epoch',
    title='Observed BCE и ranking quality')
safe_show(fig)
FIGURES['observed_bce_vs_ndcg']=fig


def get_per_user(beta, epoch):
    f = cp_files(beta,epoch)['per_user']
    require(f.is_file(),'Отсутствует per-user: '+str(f))
    data=pd.read_parquet(f)
    require(len(data)==len(USER_IDS) and not data.user_id.duplicated().any()
            and np.array_equal(data.user_id.to_numpy(dtype=np.uint64),USER_IDS)
            and int(data.eligible.sum())==2082,'Нарушен per-user протокол')
    return data

WINNER_PU=get_per_user(BEST_BETA,BEST_EPOCH)
PAIRED_ROWS=[]
# Внутри новой сетки полезно сравнить все β при одних 15 эпохах с β=1.0.
ANCHOR_NEW_PU=get_per_user(1.0,15)
if RUN_PAIRWISE_BOOTSTRAP:
    for beta in NEGATIVE_WEIGHTS:
        if beta==1.0: continue
        other=get_per_user(beta,15)
        d=bootstrap_paired_difference(other,ANCHOR_NEW_PU,metric='ndcg_at_10',
                group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        PAIRED_ROWS.append({'comparison':f'WNS β={beta:g} ep15 - WNS β=1.0 ep15',**d})

# Старый опыт читаем только если хеши, сквозной протокол и параметры совпадают.
PREVIOUS_METRICS=None
PREVIOUS_PU_15=None
PREVIOUS_PU_25=None
PREVIOUS_STATUS='Нет проверенных исходных результатов β≤0.9'
old_manifest_file=PREVIOUS_REVIEW/'selection_and_manifest.json'
old_metrics_file=PREVIOUS_REVIEW/'all_15_validation_metrics.csv'
old_per_user_file=PREVIOUS_REVIEW/'all_15_per_user.parquet'
if old_manifest_file.is_file() and old_metrics_file.is_file() and old_per_user_file.is_file():
    old_manifest=json.loads(old_manifest_file.read_text(encoding='utf-8'))
    old_compatible = (old_manifest.get('status')=='PASS_VALIDATION_ONLY'
            and old_manifest.get('data_sha256')==file_hashes
            and old_manifest.get('module_sha256')==module_hashes
            and old_manifest.get('user_id_sha256')==user_hash
            and old_manifest.get('item_id_sha256')==item_hash
            and old_manifest.get('train_rows')==len(HISTORY)
            and old_manifest.get('val_rows')==len(HOLDOUT)
            and old_manifest.get('val_eligible')==2082
            and old_manifest.get('n_users')==len(USER_IDS)
            and old_manifest.get('n_items')==len(ITEM_IDS)
            and old_manifest.get('feature_set')=='full'
            and old_manifest.get('n_factors')==N_FACTORS
            and old_manifest.get('learning_rate')==LEARNING_RATE
            and old_manifest.get('weight_decay')==WEIGHT_DECAY
            and old_manifest.get('negative_ratio')==NEGATIVE_RATIO
            and old_manifest.get('batch_size')==BATCH_SIZE
            and old_manifest.get('seed')==SEED
            and old_manifest.get('sampling_seed')==SAMPLING_SEED
            and old_manifest.get('sampling_seed')==SAMPLING_SEED)
    if not old_compatible:
        PREVIOUS_STATUS='Старые файлы присутствуют, но протокол/хеши не совпадают; сравнение пропущено'
        print(PREVIOUS_STATUS)
    else:
        old_all=pd.read_parquet(old_per_user_file)
        require({'negative_weight','epoch','user_id','eligible','ndcg_at_10','recall_at_10','ap_at_10'}<=set(old_all.columns),
                'Историческая таблица per-user: не хватает полей')
        old_all['user_id']=old_all.user_id.astype('uint64')
        old_metrics=pd.read_csv(old_metrics_file)
        require(len(old_metrics)==15 and set(old_metrics['negative_weight'].round(4))=={.2,.3,.5,.7,.9},
                'Исторические 15 метрик не совпадают с заявленной сеткой')
        for epoch in (15,25):
            part=old_all.loc[(old_all.negative_weight.round(4)==.9)&(old_all.epoch==epoch),
                             ['user_id','n_relevant','eligible','ndcg_at_10','recall_at_10','ap_at_10']].sort_values('user_id').reset_index(drop=True)
            require(len(part)==len(USER_IDS) and not part.user_id.duplicated().any()
                    and np.array_equal(part.user_id.to_numpy(dtype=np.uint64),USER_IDS)
                    and int(part.eligible.sum())==2082, 'Неполный исторический β=.9 per-user ep'+str(epoch))
            expected=old_metrics.loc[np.isclose(old_metrics.negative_weight,.9)&old_metrics.epoch.eq(epoch)].iloc[0]
            for col,summary_col in [('ndcg_at_10','eligible_ndcg_at_10'),('recall_at_10','eligible_recall_at_10'),('ap_at_10','eligible_map_at_10')]:
                require(np.isclose(float(part.loc[part.eligible,col].mean()),float(expected[summary_col]),atol=1e-10),
                        'Исторические персональные метрики противоречат общей таблице: '+col)
            if epoch==15: PREVIOUS_PU_15=part
            else: PREVIOUS_PU_25=part
        PREVIOUS_METRICS=old_metrics
        PREVIOUS_STATUS='Исторический β=.9 ep15/25 валидирован; выполняем сравнения'
        if RUN_PAIRWISE_BOOTSTRAP:
            for beta in NEGATIVE_WEIGHTS:
                d=bootstrap_paired_difference(get_per_user(beta,15),PREVIOUS_PU_15,
                        metric='ndcg_at_10',group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
                PAIRED_ROWS.append({'comparison':f'β={beta:g} ep15 - historical β=.9 ep15',**d})
            d=bootstrap_paired_difference(WINNER_PU,PREVIOUS_PU_25,
                    metric='ndcg_at_10',group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            PAIRED_ROWS.append({'comparison':'selected β>1 - historical β=.9 ep25',**d})
else:
    print('Исторический review_bundle отсутствует или неполон. Это НЕ мешает новому обучению.')
print('Исторические данные:', PREVIOUS_STATUS)

PAIRED=pd.DataFrame(PAIRED_ROWS)
if not PAIRED.empty:
    display(PAIRED[['comparison','metric','estimate','ci_low','ci_high']])

# Сравнение с предыдущими методами только через тот же evaluator и те же user_ids.
PRIOR_COMPARISONS={}
prior_files={
  'iALS binary lowrank':ROOT/'results/ials/lowrank_validation/review_bundle/winner_binary_per_user.csv',
  'ItemKNN binary normalized':ROOT/'results/item_knn/validation/winner_binary_normalized_per_user.csv',
  'EASE binary':ROOT/'results/ease/validation/review_bundle/winner_binary_per_user.csv',
}
for label,path in prior_files.items():
    if not path.is_file():
        PRIOR_COMPARISONS[label]='файл отсутствует; bootstrap пропущен'
        continue
    previous=pd.read_csv(path,dtype={'user_id':'uint64'})
    require(len(previous)==len(USER_IDS) and not previous.user_id.duplicated().any()
            and np.array_equal(np.sort(previous.user_id.to_numpy(dtype=np.uint64)),USER_IDS),
            'Прежняя модель: иной каталог пользователей')
    if RUN_PAIRWISE_BOOTSTRAP:
        d=bootstrap_paired_difference(WINNER_PU,previous,metric='ndcg_at_10',
              group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        PAIRED_ROWS.append({'comparison':f'new WNS winner - {label}',**d})
    PRIOR_COMPARISONS[label]='сравнение выполнено' if RUN_PAIRWISE_BOOTSTRAP else 'отключено'
PAIRED=pd.DataFrame(PAIRED_ROWS)
if len(PAIRED_ROWS):
    display(PAIRED[['comparison','metric','estimate','ci_low','ci_high']])
print('Предыдущие модели:',PRIOR_COMPARISONS)


REVIEW.mkdir(parents=True,exist_ok=True)
RESULTS.to_csv(REVIEW/'all_18_validation_metrics.csv',index=False)
train_rows=[]
for rec in RECORDS:
    for log in rec['training_log']:
        train_rows.append({'negative_weight':rec['config']['negative_weight'],
                           'negative_ratio':rec['config']['negative_ratio'],**log})
TRAINING_LOGS=(pd.DataFrame(train_rows).drop_duplicates(['negative_weight','epoch'])
               .sort_values(['negative_weight','epoch']).reset_index(drop=True))
require(len(TRAINING_LOGS)==len(NEGATIVE_WEIGHTS)*max(EPOCHS),
        'Восстановлены не все 150 epoch-logs')
TRAINING_LOGS.to_csv(REVIEW/'all_training_epochs.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_differences.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
WINNER_PU.to_csv(REVIEW/'winner_per_user.csv',index=False)
ANCHOR_NEW_PU.to_csv(REVIEW/'anchor_beta1_ep15_per_user.csv',index=False)
if PREVIOUS_PU_25 is not None:
    PREVIOUS_PU_25.to_csv(REVIEW/'historical_beta09_ep25_per_user.csv',index=False)
if PREVIOUS_PU_15 is not None:
    PREVIOUS_PU_15.to_csv(REVIEW/'historical_beta09_ep15_per_user.csv',index=False)
personal=[]
for beta in NEGATIVE_WEIGHTS:
    for ep in EPOCHS:
        tbl=get_per_user(beta,ep).copy()
        tbl.insert(0,'negative_weight',beta)
        tbl.insert(1,'epoch',ep)
        personal.append(tbl)
pd.concat(personal,ignore_index=True).to_parquet(
    REVIEW/'all_18_per_user.parquet',index=False,compression='zstd')
recs=pd.read_parquet(cp_files(BEST_BETA,BEST_EPOCH)['recommendations'])
require(len(recs)==TOP_K*len(USER_IDS),'Отсутствует Top-10 победителя')
recs.to_csv(REVIEW/'winner_recommendations.csv',index=False)

# β > 1 и предыдущее β <= .9, при одинаковом числе 15 эпох
current_ep15=RESULTS.loc[RESULTS.epoch.eq(15)].copy()
current_ep15['source_experiment']='beta_above_one_v1'
combined=current_ep15
HISTORY_BETA_STATUS='Исторические β≤0.9 недоступны; таблица содержит только новые β'
if PREVIOUS_METRICS is not None:
    historical=PREVIOUS_METRICS.loc[PREVIOUS_METRICS.epoch.eq(15)].copy()
    historical['source_experiment']='previous_beta_extension_v1'
    combined=pd.concat([historical,current_ep15],ignore_index=True)
    HISTORY_BETA_STATUS='Исторические β=0.2–0.9 на 15 эпохах проверены и добавлены'
# Младшие веса 0–0.1 можно добавить из ранней WNS с таким же контрольным протоколом.
initial_manifest=ROOT/'results/fm/wns_validation/review_bundle/selection_and_manifest.json'
initial_csv=ROOT/'results/fm/wns_validation/review_bundle/all_15_validation_metrics.csv'
if PREVIOUS_METRICS is not None and initial_manifest.is_file() and initial_csv.is_file():
    first_manifest=json.loads(initial_manifest.read_text(encoding='utf-8'))
    initial_ok = (first_manifest.get('data_sha256')==file_hashes
            and first_manifest.get('module_sha256')==module_hashes
            and first_manifest.get('user_id_sha256')==user_hash
            and first_manifest.get('item_id_sha256')==item_hash)
    if initial_ok:
        first=pd.read_csv(initial_csv)
        first=first.loc[first.epoch.eq(15)&first.negative_weight.lt(.2-1e-10)].copy()
        first['source_experiment']='initial_wns_v1'
        combined=pd.concat([first,combined],ignore_index=True)
        HISTORY_BETA_STATUS+='; также добавлены β<0.2'
    else:
        print('Начальная сетка β<0.2 отличается от текущего протокола и не объединяется')
combined=combined.sort_values(['negative_weight','epoch']).reset_index(drop=True)
require(not combined.duplicated(['negative_weight','epoch']).any(), 'Дубли β/epoch в объединённой таблице')
combined.to_csv(REVIEW/'combined_beta_sweep_ep15.csv',index=False)
fig=px.line(combined,x='negative_weight',y='eligible_ndcg_at_10',markers=True,
    title='WNS: полная зависимость NDCG@10 от β (15 эпох)',
    labels={'negative_weight':'Вес β','eligible_ndcg_at_10':'Eligible NDCG@10'})
safe_show(fig)
FIGURES['combined_beta_sweep_ep15']=fig

for key,fig in FIGURES.items():
    fig.write_html(str(REVIEW/f'wns_{key}.html'),include_plotlyjs='cdn',full_html=True)

MANIFEST_OUT={
    'status':'PASS_VALIDATION_ONLY','experiment':'VK-LSVD FM WNS beta above one v1.0',
    'identity_sha256':IDENTITY_HASH,'data_sha256':file_hashes,
    'module_sha256':module_hashes,'user_id_sha256':user_hash,'item_id_sha256':item_hash,
    'source_revision':REVISION['revision'],'device':EXPERIMENT_DEVICE,
    'n_users':len(USER_IDS),'n_items':len(ITEM_IDS),
    'train_rows':len(HISTORY),'val_rows':len(HOLDOUT),'val_eligible':2082,
    'feature_set':'full','n_factors':N_FACTORS,'learning_rate':LEARNING_RATE,
    'weight_decay':WEIGHT_DECAY,'batch_size':BATCH_SIZE,'seed':SEED,
    'sampling_seed':SAMPLING_SEED,'negative_ratio':NEGATIVE_RATIO,
    'negative_weights':list(NEGATIVE_WEIGHTS),'epochs':list(EPOCHS),
    'train_trajectories':len(NEGATIVE_WEIGHTS),'validation_evaluations':len(RESULTS),
    'winner':{'negative_weight':BEST_BETA,'epoch':BEST_EPOCH,
              'eligible_ndcg_at_10':float(selection.eligible_ndcg_at_10),
              'eligible_recall_at_10':float(selection.eligible_recall_at_10),
              'eligible_map_at_10':float(selection.eligible_map_at_10)},
    'history_status':HISTORY_BETA_STATUS,
    'previous_beta09_available':PREVIOUS_PU_25 is not None,
    'prior_model_paired_comparisons':PRIOR_COMPARISONS,
    'bootstrap_n':BOOTSTRAP_N,'bootstrap_seed':BOOTSTRAP_SEED,
    'bootstrap_post_selection_caveat':True,
    'note':'New β>1 independent starts; same optimizer step counts at equal epochs. Validation selection is post hoc.',
    'train_weeks':'0-24','validation_week':25,'test_read':False,
    'test_evaluated':False,'final_refit':False,
}
write_json_atomic(REVIEW/'selection_and_manifest.json',MANIFEST_OUT)
(REVIEW/'README_results.md').write_text(
    '# VK-LSVD · FM WNS β>1 validation, 10/15/25 эпох\n\n'
    'New beta grid=1.0,1.2,1.5,2.0,3.0,5.0. Six independent trajectories, '
    '18 evaluations, 150 training epochs. Fixed 3000x2000 catalog, train 0–24, '
    'validation 25 only, 64 factors, full metadata, lr=.003, AdamW decay=.1, '
    'ratio=1, seed=42. Original evaluator and labels unchanged; test never read. '
    'Contains per-user results, bootstrap, optimizer steps, concentration and graphs. '
    'Checkpoints remain in the local checkpoint_cache and are not in this bundle. '
    'Bootstrap after selection is descriptive.\n',encoding='utf-8')
require(len(RESULTS)==18 and len(TRAINING_LOGS)==150 and
        (REVIEW/'all_18_per_user.parquet').exists() and
        (REVIEW/'selection_and_manifest.json').exists(), 'Review bundle неполный')
print('PASS. Результаты:',REVIEW)
print('Результаты сохранены.')
print('Test не открывался; refit не выполнялся.')
