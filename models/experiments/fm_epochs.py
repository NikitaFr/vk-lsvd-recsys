"""fm epochs experiment on train/validation."""
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

for package in ('pyarrow', 'torch', 'plotly'):
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
OUT = ROOT/'results/fm/validation'
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
EXPECTED_EVAL_SHA256 = {
    'models/utils/evaluation/__init__.py':'e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c',
    'models/utils/evaluation/_validation.py':'6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405',
    'models/utils/evaluation/bootstrap.py':'9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221',
    'models/utils/evaluation/metrics.py':'14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938',
    'models/utils/evaluation/pipeline.py':'27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f',
    'models/utils/evaluation/ranking.py':'5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2',
    'models/utils/evaluation/relevance.py':'ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1',
}

# Только диагностическое продолжение, а не новая сетка.
TOP_K = 10
BATCH_SIZE = 8192
SEED = 42
BASELINE_EPOCH = 15
HISTORICAL_EPOCHS = (5, 10, 15)
EXTENSION_EPOCHS = (20, 25, 30, 35, 40, 45, 50)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
SOURCE_CACHE = CACHE
SOURCE_REVIEW = REVIEW
EXT_ROOT = ROOT/'results/fm/epoch_extension_15_to_50'
EXT_CACHE = EXT_ROOT/'checkpoint_cache'
EXT_REVIEW = EXT_ROOT/'review_bundle'
print('Project:',ROOT)
print('Source cache:', SOURCE_CACHE)
print('New results:', EXT_ROOT)
print('PyTorch:',torch.__version__)


file_hashes = {key:sha256_file(FILES[key]) for key in EXPECTED_FILE_SHA256}
for key, expected in EXPECTED_FILE_SHA256.items():
    require(file_hashes[key] == expected, f'Изменён {key} Parquet: {file_hashes[key]}')
module_paths = ['models/fm.py', *EXPECTED_EVAL_SHA256.keys()]
module_hashes = {path:sha256_file(ROOT/path) for path in module_paths}
require(module_hashes['models/fm.py'] == EXPECTED_FM_SHA256,
        'Контрольная сумма models/fm.py отличается от ожидаемой')
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
from models.utils.evaluation import label_relevance, evaluate_scores, bootstrap_mean, bootstrap_paired_difference
require(tuple(FEATURE_SETS) == ('id','item_meta','user_meta','full'), 'Другой контракт feature sets')
print('PASS: данные, evaluator и источник совпадают; FM SHA-256:',module_hashes['models/fm.py'])


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


prepare_start = time.perf_counter()
PREPARED = prepare_fm_data(history=HISTORY, users_metadata=users_meta,
    items_metadata=items_meta, user_ids=USER_IDS, item_ids=ITEM_IDS,
    feature_set='full', history_start_week=0, history_end_week=24)
require(len(PREPARED.labels)==1454216 and int(PREPARED.labels.sum())==532106,
        'Full FM train-target не совпадает с исходной выборкой')
print('Full features:', PREPARED.encoder.fields, '| dimensions:',PREPARED.encoder.n_features,
      '| encoding seconds:',round(time.perf_counter()-prepare_start,2))


SOURCE_IDENTITY_PATH = SOURCE_CACHE/'run_identity.json'
require(SOURCE_IDENTITY_PATH.is_file(), 'Нет исходного run_identity.json. Восстанови оригинальный checkpoint_cache из validation.zip.')
SOURCE_IDENTITY = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding='utf-8'))
SOURCE_CONTENT = SOURCE_IDENTITY['contents']
canonical_source = hashlib.sha256(json.dumps(SOURCE_CONTENT,sort_keys=True,
                                separators=(',',':')).encode()).hexdigest()
require(canonical_source==SOURCE_IDENTITY['identity_sha256'], 'Исходный cache identity повреждён')
require(SOURCE_CONTENT['file_sha256']==file_hashes and
        SOURCE_CONTENT['module_sha256']==module_hashes and
        SOURCE_CONTENT['full_fingerprint']==PREPARED.fingerprint and
        SOURCE_CONTENT['user_id_sha256']==user_hash and
        SOURCE_CONTENT['item_id_sha256']==item_hash,
        'Кэш относится к другим данным/модели/evaluator/ID или подготовленным признакам')
require(SOURCE_CONTENT['train']=={'weeks':[0,24],'rows':1454216} and
        SOURCE_CONTENT['validation']=={'week':25,'rows':16158},
        'Исходный кэш содержит другие временные границы')
require(SOURCE_CONTENT['batch_size']==8192 and SOURCE_CONTENT['seed']==42
        and SOURCE_CONTENT['top_k']==10 and SOURCE_CONTENT['device'] in ('cpu','mps'),
        'Конфигурация train/score исходного FM кэша не совпадает')
require(SOURCE_CONTENT['torch']==str(torch.__version__),
        'Версия PyTorch изменилась. Для корректного продолжения восстанови прежнее окружение: '
        +SOURCE_CONTENT['torch']+' (сейчас '+str(torch.__version__)+')')
EXPERIMENT_DEVICE=SOURCE_CONTENT['device']
require(EXPERIMENT_DEVICE=='cpu' or FM_MPS_VALIDATED,
        'В оригинальном кэше было MPS, но сейчас MPS не прошло диагностику. Не меняй backend молча.')
MODEL_CONFIG = {'n_factors':64,'learning_rate':0.003,'weight_decay':0.1,
                'batch_size':8192,'seed':42}
MODEL_STEM='full_f64_lr0.003_wd0.1_s42'


def source_files(ep):
    base = SOURCE_CACHE/f'{MODEL_STEM}_ep{ep:02d}'
    return {'pt':Path(str(base)+'.pt'),
            'marker':Path(str(base)+'_complete.json'),
            'per_user':Path(str(base)+'_per_user.parquet'),
            'recommendations':Path(str(base)+'_recommendations.parquet')}


def verify_source_epoch(ep):
    paths=source_files(ep)
    require(paths['marker'].is_file(),'Не найден исторический checkpoint marker эпохи '+str(ep))
    rec=json.loads(paths['marker'].read_text(encoding='utf-8'))
    require(rec['run_identity_sha256']==SOURCE_IDENTITY['identity_sha256']
            and rec['prepared_fingerprint']==PREPARED.fingerprint
            and rec['model_config']==MODEL_CONFIG
            and rec['feature_set']=='full' and rec['epoch']==ep,
            'Исходный checkpoint относится к другому запуску (epoch '+str(ep)+')')
    for name in ('pt','per_user','recommendations'):
        require(paths[name].is_file() and sha256_file(paths[name])==rec['sha256'][name],
                'Исходный checkpoint повреждён или отсутствует: '+str(paths[name]))
    return rec

HISTORICAL_RECORDS=[verify_source_epoch(ep) for ep in HISTORICAL_EPOCHS]
require(abs(HISTORICAL_RECORDS[-1]['eligible_ndcg_at_10']-0.0009315552245602461)<1e-12,
        'Epoch 15 не соответствует зафиксированному победителю FM')
SOURCE_PT = source_files(15)['pt']
SOURCE_PT_SHA = sha256_file(SOURCE_PT)
source_model = FactorizationMachine.load_checkpoint(SOURCE_PT,prepared=PREPARED,
                        device=EXPERIMENT_DEVICE,expected_config=MODEL_CONFIG)
require(source_model.trained_epochs==15 and len(source_model.training_log)==15,
        'Checkpoint не находится на 15-й эпохе')
print('PASS: original epoch 15 checkpoint and optimizer/RNG loaded')
print('Device:',EXPERIMENT_DEVICE,'| cached NDCG@10:',HISTORICAL_RECORDS[-1]['eligible_ndcg_at_10'])
print('Weights SHA-256:',SOURCE_PT_SHA)
del source_model
gc.collect()


EXT_CACHE.mkdir(parents=True,exist_ok=True)
EXT_REVIEW.mkdir(parents=True,exist_ok=True)
EXT_IDENTITY_CONTENT = {
    'experiment':'VK-LSVD-FM-continue-15-to-50-v1',
    'source_cache_identity_sha256':SOURCE_IDENTITY['identity_sha256'],
    'source_epoch15_weights_sha256':SOURCE_PT_SHA,
    'file_sha256':file_hashes,'module_sha256':module_hashes,
    'user_id_sha256':user_hash,'item_id_sha256':item_hash,
    'prepared_fingerprint':PREPARED.fingerprint,
    'config':MODEL_CONFIG,'device':EXPERIMENT_DEVICE,
    'source_epoch':BASELINE_EPOCH,'checkpoints':list(EXTENSION_EPOCHS),
    'top_k':TOP_K,'torch':str(torch.__version__),
    'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),
}
EXT_IDENTITY_SHA=hashlib.sha256(json.dumps(EXT_IDENTITY_CONTENT,sort_keys=True,
                               separators=(',',':')).encode()).hexdigest()
EXT_IDENTITY_FILE=EXT_CACHE/'run_identity.json'
if EXT_IDENTITY_FILE.is_file():
    require(json.loads(EXT_IDENTITY_FILE.read_text(encoding='utf-8'))==
            {'identity_sha256':EXT_IDENTITY_SHA,'contents':EXT_IDENTITY_CONTENT},
            'Этот кэш создан другим экспериментом/другой версией кода. Не смешивай результаты.')
else:
    require(not list(EXT_CACHE.iterdir()),'Новый кэш не пуст, но нет identity marker')
    write_json_atomic(EXT_IDENTITY_FILE,
                      {'identity_sha256':EXT_IDENTITY_SHA,'contents':EXT_IDENTITY_CONTENT})


def ext_files(epoch):
    base=EXT_CACHE/f'{MODEL_STEM}_ep{int(epoch):02d}'
    return {'pt':Path(str(base)+'.pt'), 'marker':Path(str(base)+'_complete.json'),
            'per_user':Path(str(base)+'_per_user.parquet'),
            'recommendations':Path(str(base)+'_recommendations.parquet')}


def eval_metrics(ev):
    return {
      'n_users':int(ev.summary['n_users']),
      'n_eligible':int(ev.summary['n_eligible']),
      'eligible_ndcg_at_10':float(ev.summary['eligible']['ndcg_at_10']),
      'eligible_recall_at_10':float(ev.summary['eligible']['recall_at_10']),
      'eligible_map_at_10':float(ev.summary['eligible']['map_at_10']),
      'all_users_ndcg_at_10':float(ev.summary['all_users']['ndcg_at_10']),
      'all_users_recall_at_10':float(ev.summary['all_users']['recall_at_10']),
      'all_users_map_at_10':float(ev.summary['all_users']['map_at_10']),
    }


def output_diagnostics(recommendations, per_user):
    counts=recommendations.groupby('item_id',sort=False).size().sort_values(ascending=False)
    hits=int(per_user['recall_at_10'].fillna(0).gt(0).sum())
    return {
        'unique_recommended_items':int(len(counts)),
        'top100_slot_share':float(counts.head(100).sum()/len(recommendations)),
        'top10_slot_share':float(counts.head(10).sum()/len(recommendations)),
        'eligible_users_with_hit':hits,
    }


def check_extension_marker(epoch):
    paths=ext_files(epoch)
    if not paths['marker'].is_file():
        return None
    rec=json.loads(paths['marker'].read_text(encoding='utf-8'))
    require(rec['extension_identity_sha256']==EXT_IDENTITY_SHA
            and rec['prepared_fingerprint']==PREPARED.fingerprint
            and rec['model_config']==MODEL_CONFIG and rec['epoch']==epoch,
            'Некорректный marker новой эпохи '+str(epoch))
    for name in ('pt','per_user','recommendations'):
        require(paths[name].is_file() and sha256_file(paths[name])==rec['sha256'][name],
                'Повреждение результата эпохи '+str(epoch)+': '+name)
    return rec


def epoch_progress(row):
    print('  epoch',row['epoch'],'BCE',f"{row['train_bce']:.6f}",
          'seconds',f"{row['seconds']:.2f}",flush=True)

print('Extension identity:',EXT_IDENTITY_SHA)
print('Source cache is read-only:',SOURCE_CACHE)


EXT_RECORDS = []
previous_checkpoint = SOURCE_PT
model = None
for epoch in EXTENSION_EPOCHS:
    paths=ext_files(epoch)
    cached=check_extension_marker(epoch)
    if cached is not None:
        print('CACHED',epoch,'NDCG@10',f"{cached['eligible_ndcg_at_10']:.6f}",flush=True)
        EXT_RECORDS.append(cached)
        previous_checkpoint=paths['pt']
        model=None
        continue
    if paths['pt'].is_file():
        # Обучение checkpoint завершилось, но запись метрик была прервана.
        model=FactorizationMachine.load_checkpoint(paths['pt'],prepared=PREPARED,
                  device=EXPERIMENT_DEVICE,expected_config=MODEL_CONFIG)
        require(model.trained_epochs==epoch,'Незавершённый checkpoint имеет не ту epoch')
        print('RECOVER saved weights, evaluating epoch',epoch,flush=True)
    else:
        if model is None:
            model=FactorizationMachine.load_checkpoint(previous_checkpoint,
                  prepared=PREPARED,device=EXPERIMENT_DEVICE,
                  expected_config=MODEL_CONFIG)
        require(model.trained_epochs<epoch and model.trained_epochs>=15,
                'Невозможная последовательность эпох; прерываем запуск')
        model.progress_callback=epoch_progress
        print('CONTINUE',model.trained_epochs,'→',epoch,flush=True)
        model.continue_fit(additional_epochs=epoch-model.trained_epochs)
        require(model.trained_epochs==epoch,'Не удалось дойти до контрольной эпохи')
        model.save_checkpoint(paths['pt'])
    require(model.trained_epochs==epoch and len(model.training_log)==epoch,
            'Неверная полная история BCE')
    scores=model.score(batch_size=BATCH_SIZE)
    require(scores.shape==(3000,2000) and np.isfinite(scores).all(),
            'Полная матрица FM score имеет неправильную форму или NaN/Inf')
    score_time=float(model.last_score_seconds)
    evaluation=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
        history=HISTORY,holdout=HOLDOUT,k=10,history_start_week=0,
        history_end_week=24,holdout_week=25)
    require(len(evaluation.per_user)==3000 and
            int(evaluation.per_user.eligible.sum())==2082 and
            len(evaluation.recommendations)==30000,
            'Evaluator вернул несовместимую пользовательскую выборку')
    require(np.array_equal(evaluation.per_user.user_id.to_numpy(dtype=np.uint64),USER_IDS),
            'Evaluator нарушил упорядочение user ID')
    evaluation.per_user.to_parquet(paths['per_user'],index=False)
    evaluation.recommendations.to_parquet(paths['recommendations'],index=False)
    log=[dict(x) for x in model.training_log]
    rec={'extension_identity_sha256':EXT_IDENTITY_SHA,
         'prepared_fingerprint':PREPARED.fingerprint,
         'model_config':MODEL_CONFIG,'feature_set':'full','epoch':epoch,
         **eval_metrics(evaluation),
         **output_diagnostics(evaluation.recommendations,evaluation.per_user),
         'train_bce_at_epoch':float(log[-1]['train_bce']),
         'fit_seconds_cumulative':float(sum(x['seconds'] for x in log)),
         'scoring_seconds':score_time,'training_log':log,
         'sha256':{k:sha256_file(paths[k]) for k in ('pt','per_user','recommendations')}}
    write_json_atomic(paths['marker'],rec)  # marker всегда последний
    EXT_RECORDS.append(rec)
    previous_checkpoint=paths['pt']
    print('PASS epoch',epoch,'NDCG',f"{rec['eligible_ndcg_at_10']:.6f}",
          'Recall',f"{rec['eligible_recall_at_10']:.6f}",
          'Unique items',rec['unique_recommended_items'],flush=True)
    del scores,evaluation
    gc.collect()
require(len(EXT_RECORDS)==7 and all(r['epoch']==e for r,e in zip(EXT_RECORDS,EXTENSION_EPOCHS)),
        'Не получены все 7 контрольных точек')
del model
gc.collect()
print('PASS: epochs 15 to 50 extended. Test never used.')


def table_row(rec):
    cols=('eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10',
          'all_users_ndcg_at_10','all_users_recall_at_10','all_users_map_at_10',
          'train_bce_at_epoch','unique_recommended_items','top100_slot_share',
          'top10_slot_share','fit_seconds_cumulative')
    return {'epoch':int(rec['epoch']),**{k:rec[k] for k in cols},
            'eligible_users_with_hit':rec.get('eligible_users_with_hit',None),
            'origin':'original' if int(rec['epoch'])<=15 else 'extension'}

ALL_RECORDS=HISTORICAL_RECORDS+EXT_RECORDS
DYNAMICS=pd.DataFrame([table_row(r) for r in ALL_RECORDS]).sort_values('epoch').reset_index(drop=True)
require(DYNAMICS.epoch.tolist()==[5,10,15,20,25,30,35,40,45,50],
        'Неправильная последовательность точек')
HISTORY_LOG=pd.DataFrame(EXT_RECORDS[-1]['training_log'])
require(HISTORY_LOG.epoch.tolist()==list(range(1,51)),
        'Потеряна история BCE или выполнен новый fit вместо continue_fit')
display(DYNAMICS[['epoch','eligible_ndcg_at_10','eligible_recall_at_10',
                  'eligible_map_at_10','train_bce_at_epoch','unique_recommended_items',
                  'top100_slot_share']])
BASE_NDCG=float(DYNAMICS.loc[DYNAMICS.epoch.eq(15),'eligible_ndcg_at_10'].iloc[0])
LATEST_NDCG=float(DYNAMICS.loc[DYNAMICS.epoch.eq(50),'eligible_ndcg_at_10'].iloc[0])
PEAK=DYNAMICS.sort_values(['eligible_ndcg_at_10','epoch'],ascending=[False,True]).iloc[0]
print('At epoch 15: NDCG',f'{BASE_NDCG:.8f}')
print('At epoch 50: NDCG',f'{LATEST_NDCG:.8f}')
print('Highest of measured epochs:',int(PEAK.epoch),'NDCG',f"{PEAK['eligible_ndcg_at_10']:.8f}")
print('Historical iALS reference: NDCG@10 ≈ 0.020702 (separate validation result)')


BASE_PER_USER=pd.read_parquet(source_files(15)['per_user'])
require(len(BASE_PER_USER)==3000 and
        np.array_equal(BASE_PER_USER.user_id.to_numpy(dtype=np.uint64),USER_IDS),
        'Baseline epoch 15 per_user не соответствует train каталогу')
PAIRED_ROWS=[]
for r in EXT_RECORDS:
    path=ext_files(r['epoch'])['per_user']
    pu=pd.read_parquet(path)
    require(len(pu)==3000 and np.array_equal(pu.user_id.to_numpy(dtype=np.uint64),USER_IDS),
            'Разные user ID на checkpoint '+str(r['epoch']))
    for m in ('ndcg_at_10','recall_at_10','ap_at_10'):
        stats=bootstrap_paired_difference(pu,BASE_PER_USER,metric=m,group='eligible',
                      n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        PAIRED_ROWS.append({'epoch':r['epoch'],'comparison':'extended - epoch15',**stats})
PAIRED=pd.DataFrame(PAIRED_ROWS)
display(PAIRED.loc[PAIRED.metric.eq('ndcg_at_10'),
       ['epoch','estimate','ci_low','ci_high']])


FIGURES={}

def safe_show(fig):
    try:
        fig.show()
    except (ValueError, ImportError) as exc:
        print('Inline Plotly недоступен:',exc,'; график будет сохранён как HTML.')

fig=px.line(DYNAMICS,x='epoch',y='eligible_ndcg_at_10',markers=True,
    title='FM Full · NDCG@10 при продолжении 15 → 50 эпох',
    labels={'epoch':'Эпоха','eligible_ndcg_at_10':'Eligible NDCG@10'})
# Не проводим линию iALS на том же масштабе: она сплющит динамику FM.
FIGURES['ndcg']=fig
safe_show(fig)

fig=px.line(DYNAMICS,x='epoch',y=['eligible_recall_at_10','eligible_map_at_10'],
    markers=True,title='FM Full · Recall@10 и MAP@10',
    labels={'epoch':'Эпоха','value':'Eligible metric','variable':'Метрика'})
FIGURES['recall_map']=fig
safe_show(fig)

fig=px.line(HISTORY_LOG,x='epoch',y='train_bce',title='FM Full · Train BCE по каждой эпохе',
    labels={'epoch':'Эпоха','train_bce':'BCE на наблюдавшихся парах'})
FIGURES['bce']=fig
safe_show(fig)

fig=px.line(DYNAMICS,x='epoch',y='top100_slot_share',markers=True,
    title='FM Full · Доля Top-100 самых часто рекомендуемых видео',
    labels={'epoch':'Эпоха','top100_slot_share':'Доля 30 000 слотов'})
FIGURES['concentration']=fig
safe_show(fig)


EXT_REVIEW.mkdir(parents=True,exist_ok=True)
DYNAMICS.to_csv(EXT_REVIEW/'epoch_05_to_50_metrics.csv',index=False)
HISTORY_LOG.to_csv(EXT_REVIEW/'train_bce_epochs_01_to_50.csv',index=False)
PAIRED.to_csv(EXT_REVIEW/'paired_delta_vs_epoch15.csv',index=False)
for r in EXT_RECORDS:
    # Сохранение персональных метрик для сравнения checkpoints.
    ep=int(r['epoch'])
    pu=pd.read_parquet(ext_files(ep)['per_user'])
    pu.to_parquet(EXT_REVIEW/f'epoch_{ep:02d}_per_user.parquet',index=False,compression='zstd')

# Выдачи на измеренной лучшей эпохе — для диагностики перекоса популярности.
peak_epoch=int(PEAK.epoch)
peak_path=(source_files(peak_epoch) if peak_epoch<=15 else ext_files(peak_epoch))
peak_recs=pd.read_parquet(peak_path['recommendations'])
require(len(peak_recs)==30000,'Неполные рекомендации измеренного лучшего checkpoint')
peak_recs.to_csv(EXT_REVIEW/'best_measured_epoch_recommendations.csv',index=False)

for name,fig in FIGURES.items():
    fig.write_html(str(EXT_REVIEW/f'fm_extended_{name}.html'),
                   include_plotlyjs='cdn',full_html=True)

REPORT={'status':'PASS_VALIDATION_ONLY','experiment':'FM epoch extension 15 to 50',
  'source_cache_identity_sha256':SOURCE_IDENTITY['identity_sha256'],
  'source_epoch15_weights_sha256':SOURCE_PT_SHA,
  'extension_identity_sha256':EXT_IDENTITY_SHA,
  'files_sha256':file_hashes,'module_sha256':module_hashes,
  'prepared_fingerprint':PREPARED.fingerprint,
  'source_checkpoint_epoch':15,'source_checkpoint_metrics':table_row(HISTORICAL_RECORDS[-1]),
  'config':MODEL_CONFIG,'device':EXPERIMENT_DEVICE,
  'epochs_scored':DYNAMICS.epoch.astype(int).tolist(),
  'new_training_epochs':list(range(16,51)),
  'best_measured_epoch':peak_epoch,
  'best_measured_ndcg':float(PEAK['eligible_ndcg_at_10']),
  'epoch50_ndcg':LATEST_NDCG,
  'fixed_users':len(USER_IDS),'fixed_items':len(ITEM_IDS),
  'train_weeks':[0,24],'validation_week':25,
  'test_read':False,'final_refit':False,'post_selection_caveat':True}
write_json_atomic(EXT_REVIEW/'extension_manifest.json',REPORT)
(EXT_REVIEW/'README_results.md').write_text(
  '# FM continuation 15–50 epochs · validation-only\n\n'
  'One fixed model: full metadata, f=64, lr=0.003, AdamW decay=0.1, seed 42, batch 8192.\n'
  'Epochs 5/10/15 taken from prior run; 20/25/30/35/40/45/50 from continuation.\n'
  'Fixed train 0–24, validation week 25; eligible NDCG@10, Recall@10, MAP@10.\n'
  'Per-user paired CIs are descriptive after checkpoint selection. No test use.\n',
  encoding='utf-8')
require(all((EXT_REVIEW/name).is_file() for name in
            ['epoch_05_to_50_metrics.csv','train_bce_epochs_01_to_50.csv',
             'paired_delta_vs_epoch15.csv','extension_manifest.json',
             'best_measured_epoch_recommendations.csv']),
        'Review bundle неполный')
print('PASS. Report:',EXT_REVIEW)
print('Send a ZIP containing only review_bundle, not checkpoint_cache.')
print('Original cache unchanged:',SOURCE_CACHE)
print('No test was read.')
