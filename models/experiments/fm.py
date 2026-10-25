"""fm experiment on train/validation."""
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
FACTORS = (4,8,16,32,64)
LEARNING_RATES = (.0003,.001,.003)
WEIGHT_DECAYS = (0.,.01,.1)
EPOCHS = (5,10,15)
GRID = tuple(product(FACTORS, LEARNING_RATES, WEIGHT_DECAYS))
TOP_K = 10
BATCH_SIZE = 8192
SEED = 42
ROBUST_SEEDS = (42,43,44)
BOOTSTRAP_N, BOOTSTRAP_SEED = 2000, 42
RUN_BENCHMARK = True
RUN_FULL_GRID = True
RUN_ABLATION = True
RUN_SEEDS = True
EXPERIMENT_DEVICE = 'mps' if FM_MPS_VALIDATED else 'cpu'
# На Mac можно вручную выбрать 'cpu' / 'mps' ДО первого полного запуска.
require(len(GRID) == 45 and len(set(GRID)) == 45 and EPOCHS == (5,10,15),
        'Неправильная утверждённая FM-сетка')
print('Project:', ROOT)
print('Torch:', torch.__version__, '| MPS validated by subprocess:', FM_MPS_VALIDATED,
      '| выбранное устройство:', EXPERIMENT_DEVICE)


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
tiny_m = FactorizationMachine(n_factors=4,batch_size=2,seed=42,device='cpu')
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
print('PASS: synthetic FM + official evaluator; huge uint64 and rel=0 mask')
del tiny_prepared,tiny_m,tiny_scores,tiny_eval
_ = gc.collect()


PREPARED = {}
prepare_seconds = {}
start = time.perf_counter()
PREPARED['full'] = prepare_fm_data(history=HISTORY,users_metadata=users_meta,
    items_metadata=items_meta,user_ids=USER_IDS,item_ids=ITEM_IDS,feature_set='full',
    history_start_week=0,history_end_week=24)
prepare_seconds['full'] = time.perf_counter()-start
require(len(PREPARED['full'].labels)==1454216
        and int(PREPARED['full'].labels.sum())==532106
        and PREPARED['full'].encoder.n_features>5000,
        'FM prepared неверно кодирует train')
print('FM full: D =',PREPARED['full'].encoder.n_features,
      '| active fields =',PREPARED['full'].encoder.fields,
      '| prepare time =',round(prepare_seconds['full'],2),'sec')
BENCHMARK_ROWS=[]
if RUN_BENCHMARK:
    for dev in ('cpu','mps'):
        if dev=='mps' and not FM_MPS_VALIDATED:
            continue
        print('Benchmark (1 epoch + full score):',dev,flush=True)
        b = FactorizationMachine(n_factors=16,learning_rate=.001,weight_decay=.01,
                                 batch_size=BATCH_SIZE,seed=42,device=dev)
        started=time.perf_counter()
        b.fit_prepared(PREPARED['full'],epochs=1)
        fit_sec=time.perf_counter()-started
        s=b.score(batch_size=BATCH_SIZE)
        require(s.shape==(3000,2000) and np.isfinite(s).all(), 'Benchmark score неверен')
        BENCHMARK_ROWS.append(dict(device=dev,n_factors=16,fit_one_epoch_seconds=fit_sec,
                                   score_full_seconds=b.last_score_seconds,
                                   n_train=len(HISTORY),n_scored_pairs=s.size))
        del b,s
        _ = gc.collect()
BENCHMARK=pd.DataFrame(BENCHMARK_ROWS)
display(BENCHMARK)
print('Для всей сетки выбрано:',EXPERIMENT_DEVICE)


CACHE.mkdir(parents=True,exist_ok=True)
REVIEW.mkdir(parents=True,exist_ok=True)
FINGERPRINTS = {'full':PREPARED['full'].fingerprint}
RUN_IDENTITY = {
    'experiment':'VK-LSVD-FM-validation-v1',
    'revision':REVISION['revision'],'file_sha256':file_hashes,'module_sha256':module_hashes,
    'user_id_sha256':user_hash,'item_id_sha256':item_hash,
    'train':{'weeks':[0,24],'rows':len(HISTORY)},'validation':{'week':25,'rows':len(HOLDOUT)},
    'fm_feature_sets':{k:list(v) for k,v in FEATURE_SETS.items()},
    'full_fingerprint':FINGERPRINTS['full'],
    'grid':list(map(list,GRID)),'epochs':list(EPOCHS),
    'batch_size':BATCH_SIZE,'seed':SEED,'device':EXPERIMENT_DEVICE,
    'top_k':TOP_K,'torch':str(torch.__version__),'numpy':np.__version__,
    'pandas':pd.__version__,'python':platform.python_version(),
}
IDENTITY_HASH = hashlib.sha256(json.dumps(RUN_IDENTITY,sort_keys=True,
                             separators=(',',':')).encode()).hexdigest()
identity_path=CACHE/'run_identity.json'
if identity_path.exists():
    previous=json.loads(identity_path.read_text(encoding='utf-8'))
    require(previous=={'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY},
            'Кэш относится к ДРУГОМУ запуску! Не удаляй его без резервной копии; '
            'используй другой OUT или восстанови прежние файлы/настройки.')
else:
    require(not list(CACHE.iterdir()),
            'Кэш не пуст, но run_identity.json отсутствует. Остановись: нельзя доверять результатам.')
    write_json_atomic(identity_path,{'identity_sha256':IDENTITY_HASH,'contents':RUN_IDENTITY})


def config_stem(feature_set,f,lr,decay,seed):
    return f'{feature_set}_f{int(f)}_lr{float(lr):.4g}_wd{float(decay):.4g}_s{int(seed)}'


def cp_files(stem,epoch):
    base=CACHE/f'{stem}_ep{int(epoch):02d}'
    return {'pt':Path(f'{base}.pt'),'per_user':Path(f'{base}_per_user.parquet'),
            'recommendations':Path(f'{base}_recommendations.parquet'),
            'marker':Path(f'{base}_complete.json')}


def as_metrics(ev):
    return {'n_users':ev.summary['n_users'],'n_eligible':ev.summary['n_eligible'],
            'eligible_ndcg_at_10':ev.summary['eligible']['ndcg_at_10'],
            'eligible_recall_at_10':ev.summary['eligible']['recall_at_10'],
            'eligible_map_at_10':ev.summary['eligible']['map_at_10'],
            'all_users_ndcg_at_10':ev.summary['all_users']['ndcg_at_10'],
            'all_users_recall_at_10':ev.summary['all_users']['recall_at_10'],
            'all_users_map_at_10':ev.summary['all_users']['map_at_10']}


def rank_diagnostics(recs):
    require(len(recs)==len(USER_IDS)*TOP_K,'Не все Top-10 сохранены')
    counts=recs.groupby('item_id',sort=False).size().sort_values(ascending=False)
    return {'unique_recommended_items':int(len(counts)),
            'top100_slot_share':float(counts.head(100).sum()/len(recs)),
            'top10_slot_share':float(counts.head(10).sum()/len(recs))}


def config_dict(f,lr,decay,seed):
    return {'n_factors':int(f),'learning_rate':float(lr),
            'weight_decay':float(decay),'batch_size':int(BATCH_SIZE),'seed':int(seed)}


def completed_record(files, feature_set, f, lr, decay, seed, epoch, fingerprint):
    if not files['marker'].is_file():
        return None
    rec=json.loads(files['marker'].read_text(encoding='utf-8'))
    require(rec['run_identity_sha256']==IDENTITY_HASH
            and rec['prepared_fingerprint']==fingerprint
            and rec['feature_set']==feature_set
            and rec['model_config']==config_dict(f,lr,decay,seed)
            and rec['epoch']==int(epoch), 'Недействительный checkpoint marker')
    for name in ('pt','per_user','recommendations'):
        require(files[name].is_file() and sha256_file(files[name])==rec['sha256'][name],
                'Данные завершённого checkpoint повреждены: '+str(files[name]))
    return rec


def progress(row):
    print('   epoch',row['epoch'],'BCE=',f"{row['train_bce']:.5f}",
          'time=',f"{row['seconds']:.1f}s",flush=True)


def run_training(feature_set,f,lr,decay,seed,stops,prepared):
    stops=tuple(sorted(set(int(x) for x in stops)))
    require(stops and stops[0]>0,'Неверный набор epochs')
    stem=config_stem(feature_set,f,lr,decay,seed)
    if feature_set not in FINGERPRINTS:
        FINGERPRINTS[feature_set]=prepared.fingerprint
    fingerprint=FINGERPRINTS[feature_set]
    model=None
    records=[]
    cfg=config_dict(f,lr,decay,seed)
    for epoch in stops:
        paths=cp_files(stem,epoch)
        saved=completed_record(paths,feature_set,f,lr,decay,seed,epoch,fingerprint)
        if saved is not None:
            print('   CACHED',stem,'epoch',epoch,flush=True)
            records.append(saved)
            # Если предыдущая точка восстановлена из кэша, продолжать именно её веса.
            model=None
            continue
        if paths['pt'].exists():
            # Весовые состояния уже записаны: дочитываем ТОЧНО этот epoch.
            model=FactorizationMachine.load_checkpoint(paths['pt'],prepared=prepared,
                                device=EXPERIMENT_DEVICE,expected_config=cfg)
        else:
            if model is None:
                previous=[(int(r['epoch']),cp_files(stem,int(r['epoch']))['pt'])
                          for r in records if int(r['epoch'])<epoch]
                if previous:
                    _,previous_pt=max(previous,key=lambda p:p[0])
                    model=FactorizationMachine.load_checkpoint(previous_pt,prepared=prepared,
                                device=EXPERIMENT_DEVICE,expected_config=cfg)
                else:
                    model=FactorizationMachine(**cfg,device=EXPERIMENT_DEVICE,
                                               progress_callback=progress)
                    model.fit_prepared(prepared,epochs=epoch)
            model.progress_callback=progress
            if model.trained_epochs < epoch:
                model.continue_fit(additional_epochs=epoch-model.trained_epochs)
            require(model.trained_epochs == epoch, 'Неправильная epoch перед checkpoint')
            model.save_checkpoint(paths['pt'])  # атомарная запись в FM API
        require(model.trained_epochs==epoch,'Некорректное восстановление checkpoint')
        model.progress_callback=progress
        print('   VALIDATION',stem,'epoch',epoch,flush=True)
        t0=time.perf_counter()
        scores=model.score(batch_size=BATCH_SIZE)
        score_seconds=float(model.last_score_seconds)
        require(scores.shape==(len(USER_IDS),len(ITEM_IDS)) and np.isfinite(scores).all(),
                'Некорректные FM scores')
        ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
                history=HISTORY,holdout=HOLDOUT,k=TOP_K,
                history_start_week=0,history_end_week=24,holdout_week=25)
        eval_seconds=time.perf_counter()-t0-score_seconds
        require(len(ev.per_user)==len(USER_IDS) and int(ev.per_user.eligible.sum())==2082
                and len(ev.recommendations)==len(USER_IDS)*TOP_K,
                'Неверный размер результата evaluator')
        require(np.array_equal(ev.per_user.user_id.to_numpy(dtype=np.uint64),USER_IDS),
                'Evaluator переставил user ID')
        ev.per_user.to_parquet(paths['per_user'],index=False)
        ev.recommendations.to_parquet(paths['recommendations'],index=False)
        current_log=[dict(x) for x in model.training_log]
        result={'run_identity_sha256':IDENTITY_HASH,'prepared_fingerprint':fingerprint,
                'feature_set':feature_set,'model_config':cfg,'epoch':epoch,
                **as_metrics(ev),**rank_diagnostics(ev.recommendations),
                'fit_seconds_cumulative':float(sum(v['seconds'] for v in current_log)),
                'scoring_seconds':score_seconds,'evaluator_seconds':max(0.,eval_seconds),
                'train_bce_at_epoch':float(current_log[-1]['train_bce']),
                'training_log':current_log,
                'sha256':{k:sha256_file(paths[k]) for k in ('pt','per_user','recommendations')}}
        write_json_atomic(paths['marker'],result)  # строго последняя операция
        print('   PASS',stem,'ep',epoch,'NDCG=',f"{result['eligible_ndcg_at_10']:.6f}",
              'Recall=',f"{result['eligible_recall_at_10']:.6f}",flush=True)
        records.append(result)
        del scores,ev
        _=gc.collect()
    del model
    _=gc.collect()
    return records


def result_row(rec):
    return {'feature_set':rec['feature_set'],**rec['model_config'],
            'epoch':rec['epoch'],
            **{k:rec[k] for k in ('eligible_ndcg_at_10','eligible_recall_at_10',
                'eligible_map_at_10','all_users_ndcg_at_10','all_users_recall_at_10',
                'all_users_map_at_10','n_users','n_eligible','fit_seconds_cumulative',
                'scoring_seconds','evaluator_seconds','train_bce_at_epoch',
                'unique_recommended_items','top100_slot_share','top10_slot_share')}}

print('PASS cache identity:',IDENTITY_HASH)


GRID_RECORDS=[]
if RUN_FULL_GRID:
    for j,(f,lr,decay) in enumerate(GRID,1):
        print(f'\n===== FM grid {j}/{len(GRID)}: f={f}, lr={lr}, decay={decay} =====',flush=True)
        GRID_RECORDS += run_training('full',f,lr,decay,SEED,EPOCHS,PREPARED['full'])
    RESULTS=pd.DataFrame([result_row(r) for r in GRID_RECORDS])
    require(len(RESULTS)==len(GRID)*len(EPOCHS) and
            not RESULTS.duplicated(['feature_set','n_factors','learning_rate',
                                    'weight_decay','seed','epoch']).any(),
            'Не все 135 checkpoints завершены')
    require((RESULTS.n_users==3000).all() and (RESULTS.n_eligible==2082).all(),
            'Разные наборы пользователей')
    # Tie break закреплён ДО просмотра результатов.
    SELECTION_COLS=['eligible_ndcg_at_10','n_factors','epoch','learning_rate','weight_decay']
    SELECTED=RESULTS.sort_values(SELECTION_COLS,ascending=[False,True,True,True,True],
                                  kind='mergesort').reset_index(drop=True)
    BEST=SELECTED.iloc[0]
    # Разные траектории: две лучшие по максимуму на своих checkpoint.
    TOP_TWO=SELECTED.drop_duplicates(['n_factors','learning_rate','weight_decay']).head(2).copy()
    require(len(TOP_TWO)==2,'Нужно две отличные траектории для robustness')
    display(SELECTED.head(15)[['n_factors','learning_rate','weight_decay','epoch',
               'eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10',
               'fit_seconds_cumulative','unique_recommended_items']])
    print('BEST FM:',BEST[SELECTION_COLS].to_dict())
else:
    print('RUN_FULL_GRID=False: дальнейшие ablation, seeds и итоговый выбор отключены.')


FIGURES={}
if RUN_FULL_GRID:
    plot=RESULTS.copy()
    plot['learning_rate_label']=plot.learning_rate.map(lambda x:f'{x:g}')
    plot['weight_decay_label']=plot.weight_decay.map(lambda x:f'{x:g}')
    plot['epoch_label']=plot.epoch.astype(str)
    for metric,label in [('eligible_ndcg_at_10','NDCG@10'),
                         ('eligible_recall_at_10','Recall@10'),
                         ('eligible_map_at_10','MAP@10')]:
        fig=px.line(plot.sort_values('n_factors'),x='n_factors',y=metric,
             color='learning_rate_label',line_dash='weight_decay_label',
             facet_col='epoch_label',markers=True,
             hover_data=['epoch','train_bce_at_epoch','fit_seconds_cumulative'],
             title=f'FM · Validation Eligible {label} (full, 135 checkpoints)',
             labels={'n_factors':'Количество факторов f',metric:label,
                     'learning_rate_label':'Learning rate',
                     'weight_decay_label':'AdamW decay','epoch_label':'Epoch'})
        fig.update_xaxes(type='category')
        fig.update_layout(height=540)
        fig.show()
        FIGURES[metric]=fig
    fig=px.scatter(RESULTS,x='train_bce_at_epoch',y='eligible_ndcg_at_10',
                   color=RESULTS.epoch.astype(str),symbol='n_factors',
                   hover_data=['learning_rate','weight_decay'],
                   title='FM: обучающая BCE и ранжирующая NDCG — разные задачи',
                   labels={'x':'Train BCE','eligible_ndcg_at_10':'Eligible NDCG@10',
                           'color':'Epoch'})
    fig.show()
    FIGURES['train_bce_vs_ndcg']=fig


ABLATION_RECORDS=[]
if RUN_FULL_GRID and RUN_ABLATION:
    for feature_set in ('id','item_meta','user_meta'):
        print('Preparing ablation',feature_set,flush=True)
        start=time.perf_counter()
        PREPARED[feature_set]=prepare_fm_data(history=HISTORY,
            users_metadata=users_meta,items_metadata=items_meta,
            user_ids=USER_IDS,item_ids=ITEM_IDS,feature_set=feature_set,
            history_start_week=0,history_end_week=24)
        prepare_seconds[feature_set]=time.perf_counter()-start
        ABLATION_RECORDS += run_training(feature_set,int(BEST.n_factors),
            float(BEST.learning_rate),float(BEST.weight_decay),SEED,
            (int(BEST.epoch),),PREPARED[feature_set])
    full_match=[r for r in GRID_RECORDS
        if r['model_config']==config_dict(int(BEST.n_factors),float(BEST.learning_rate),
                                          float(BEST.weight_decay),SEED)
        and r['epoch']==int(BEST.epoch)]
    require(len(full_match)==1,'Невозможно найти full winner')
    ABLATION_RECORDS.append(full_match[0])
    ABLATION=pd.DataFrame([result_row(r) for r in ABLATION_RECORDS])
    require(set(ABLATION.feature_set)==set(FEATURE_SETS),'Не все четыре ablation выполнены')
    ABLATION['D']=ABLATION.feature_set.map({k:p.encoder.n_features for k,p in PREPARED.items()})
    display(ABLATION[['feature_set','D','eligible_ndcg_at_10',
                      'eligible_recall_at_10','eligible_map_at_10',
                      'unique_recommended_items','top100_slot_share']])
    fig=px.bar(ABLATION,x='feature_set',y='eligible_ndcg_at_10',text='eligible_ndcg_at_10',
               title='FM ablation: фиксированные параметры, возрастающий набор признаков')
    fig.update_traces(texttemplate='%{text:.6f}',textposition='outside')
    fig.show()
    FIGURES['ablation_ndcg']=fig


SEED_RECORDS=[]
if RUN_FULL_GRID and RUN_SEEDS:
    for j,(_,candidate) in enumerate(TOP_TWO.iterrows(),1):
        f,lr,decay,ep=(int(candidate.n_factors),float(candidate.learning_rate),
                       float(candidate.weight_decay),int(candidate.epoch))
        rec42=[r for r in GRID_RECORDS if
               r['model_config']==config_dict(f,lr,decay,42) and r['epoch']==ep]
        require(len(rec42)==1,'Отсутствует seed 42 в основной сетке')
        SEED_RECORDS.extend(rec42)
        for seed in (43,44):
            print(f'Robustness {j}/2, seed={seed}',flush=True)
            SEED_RECORDS += run_training('full',f,lr,decay,seed,(ep,),PREPARED['full'])
    ROBUSTNESS=pd.DataFrame([{'candidate':j+1,**result_row(r)}
         for j in range(2) for r in SEED_RECORDS[j*3:(j+1)*3]])
    require(len(ROBUSTNESS)==6 and set(ROBUSTNESS.seed)==set(ROBUST_SEEDS),
            'Неполный robustness анализ')
    display(ROBUSTNESS[['candidate','n_factors','learning_rate','weight_decay',
                        'epoch','seed','eligible_ndcg_at_10','eligible_recall_at_10']])
    fig=px.line(ROBUSTNESS.sort_values('seed'),x='seed',y='eligible_ndcg_at_10',
                color='candidate',markers=True,
                title='FM: чувствительность двух выбранных траекторий к seed')
    fig.show()
    FIGURES['seed_robustness']=fig


def per_user_for_record(record):
    cfg=record['model_config']
    path=cp_files(config_stem(record['feature_set'],cfg['n_factors'],cfg['learning_rate'],
                  cfg['weight_decay'],cfg['seed']),record['epoch'])['per_user']
    require(path.is_file(),'Нет сохранённого per-user '+str(path))
    frame=pd.read_parquet(path)
    require(len(frame)==len(USER_IDS) and
            np.array_equal(frame.user_id.to_numpy(dtype=np.uint64),USER_IDS),
            'Разные ID/order в per-user')
    return frame

BOOTSTRAP_ROWS=[]
PAIRED_ROWS=[]
PRIOR_STATUS={}
if RUN_FULL_GRID:
    best_rec=next(r for r in GRID_RECORDS if
       r['model_config']==config_dict(int(BEST.n_factors),float(BEST.learning_rate),
                                      float(BEST.weight_decay),42)
       and r['epoch']==int(BEST.epoch))
    best_pu=per_user_for_record(best_rec)
    for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
        for group in ('eligible','all_users'):
            interval=bootstrap_mean(best_pu,metric=metric,group=group,
                                    n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED,ci=.95)
            BOOTSTRAP_ROWS.append({'configuration':'FM best full','metric':metric,
                                   'group':group,**interval})
    if RUN_ABLATION:
        for record in ABLATION_RECORDS:
            label=record['feature_set']
            if label=='full':
                continue
            pu=per_user_for_record(record)
            diff=bootstrap_paired_difference(best_pu,pu,metric='ndcg_at_10',
                      group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            PAIRED_ROWS.append({'comparison':f'FM full - FM {label}',**diff})
    prior_files={
       'ItemKNN binary normalized': ROOT/'results/item_knn/validation/winner_binary_normalized_per_user.csv',
       'iALS binary lowrank': ROOT/'results/ials/lowrank_validation/review_bundle/winner_binary_per_user.csv',
       'EASE binary': ROOT/'results/ease/validation/review_bundle/winner_binary_per_user.csv',
    }
    for label,path in prior_files.items():
        if not path.is_file():
            PRIOR_STATUS[label]='не найдено: парное сравнение пропущено'
            continue
        previous=pd.read_csv(path,dtype={'user_id':'uint64'})
        require(len(previous)==len(USER_IDS) and not previous.user_id.duplicated().any()
                and np.array_equal(np.sort(previous.user_id.to_numpy(dtype=np.uint64)),USER_IDS),
                'Сохранённый результат '+label+' использует другой набор ID')
        for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
            delta=bootstrap_paired_difference(best_pu,previous,metric=metric,
                      group='eligible',n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            PAIRED_ROWS.append({'comparison':f'FM full - {label}',**delta})
        PRIOR_STATUS[label]='парный bootstrap рассчитан'
    BOOTSTRAP=pd.DataFrame(BOOTSTRAP_ROWS)
    PAIRED=pd.DataFrame(PAIRED_ROWS)
    display(BOOTSTRAP[['metric','group','estimate','ci_low','ci_high']])
    if not PAIRED.empty:
        display(PAIRED[['comparison','metric','group','estimate','ci_low','ci_high']])
    print('Ранее обученные модели:',PRIOR_STATUS)


if RUN_FULL_GRID:
    REVIEW.mkdir(parents=True,exist_ok=True)
    RESULTS.sort_values(['n_factors','learning_rate','weight_decay','epoch']).to_csv(
        REVIEW/'all_135_validation_metrics.csv',index=False)
    TOP_TWO.to_csv(REVIEW/'top2_distinct_trajectories.csv',index=False)
    all_users_parts=[]
    for record in GRID_RECORDS:
        part=per_user_for_record(record)
        cfg=record['model_config']
        part.insert(0,'n_factors',cfg['n_factors'])
        part.insert(1,'learning_rate',cfg['learning_rate'])
        part.insert(2,'weight_decay',cfg['weight_decay'])
        part.insert(3,'epoch',record['epoch'])
        all_users_parts.append(part)
    all_users=pd.concat(all_users_parts,ignore_index=True)
    require(len(all_users)==len(GRID)*len(EPOCHS)*len(USER_IDS),
            'Не все персональные результаты основной сетки объединены')
    all_users.to_parquet(REVIEW/'all_135_per_user.parquet',index=False,compression='zstd')
    del all_users,all_users_parts
    best_pu.to_csv(REVIEW/'winner_full_per_user.csv',index=False)
    winner_files=cp_files(config_stem('full',int(BEST.n_factors),float(BEST.learning_rate),
                 float(BEST.weight_decay),42),int(BEST.epoch))
    best_recs=pd.read_parquet(winner_files['recommendations'])
    require(len(best_recs)==30000,'Потеряны Top-10 победителя')
    best_recs.to_csv(REVIEW/'winner_full_recommendations.csv',index=False)
    training_rows=[]
    for record in GRID_RECORDS:
        cfg=record['model_config']
        for log in record['training_log']:
            training_rows.append({'feature_set':'full',**cfg,**log})
    train_logs=(pd.DataFrame(training_rows).drop_duplicates(
        ['feature_set','n_factors','learning_rate','weight_decay','seed','epoch'])
        .sort_values(['n_factors','learning_rate','weight_decay','epoch']))
    require(len(train_logs)==len(GRID)*max(EPOCHS),'Неполная история 675 эпох')
    train_logs.to_csv(REVIEW/'all_training_epochs.csv',index=False)
    BOOTSTRAP.to_csv(REVIEW/'bootstrap_winner.csv',index=False)
    PAIRED.to_csv(REVIEW/'paired_differences.csv',index=False)
    label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
    if not BENCHMARK.empty:
        BENCHMARK.to_csv(REVIEW/'cpu_mps_benchmark.csv',index=False)
    if RUN_ABLATION:
        ABLATION.to_csv(REVIEW/'ablation_4_feature_sets.csv',index=False)
        for record in ABLATION_RECORDS:
            per_user_for_record(record).to_parquet(
                REVIEW/f"ablation_{record['feature_set']}_per_user.parquet",index=False)
    if RUN_SEEDS:
        ROBUSTNESS.to_csv(REVIEW/'seed_robustness_2x3.csv',index=False)
        for j,record in enumerate(SEED_RECORDS):
            per_user_for_record(record).to_parquet(
                REVIEW/f'robustness_{j+1}_per_user.parquet',index=False)
    for label,fig in FIGURES.items():
        fig.write_html(str(REVIEW/f'fm_{label}.html'),include_plotlyjs='cdn',full_html=True)
    selection={'metric':'Eligible NDCG@10, linear gains 0/1/2',
               'tie_break':'Higher NDCG, then smaller factors, epoch, lr, decay',
               'winner':BEST[['feature_set','n_factors','learning_rate','weight_decay',
                              'epoch','eligible_ndcg_at_10','eligible_recall_at_10',
                              'eligible_map_at_10']].to_dict(),
               'two_distinct_trajectories':TOP_TWO[['n_factors','learning_rate',
                   'weight_decay','epoch','eligible_ndcg_at_10']].to_dict(orient='records')}
    MANIFEST_OUT={'status':'PASS_VALIDATION_ONLY',
       'experiment':'VK-LSVD FM validation v1',
       'run_identity_sha256':IDENTITY_HASH,'source_revision':REVISION['revision'],
       'files_sha256':file_hashes,'module_sha256':module_hashes,
       'user_id_sha256':user_hash,'item_id_sha256':item_hash,
       'device':EXPERIMENT_DEVICE,'n_users':3000,'n_items':2000,
       'train_rows':len(HISTORY),'validation_rows':len(HOLDOUT),'eligible_users':2082,
       'main_grid_trajectories':len(GRID),'main_grid_evaluations':len(RESULTS),
       'ablation_completed':bool(RUN_ABLATION),'seed_robustness_completed':bool(RUN_SEEDS),
       'benchmark_completed':bool(RUN_BENCHMARK),
       'feature_prepare_seconds':prepare_seconds,'prior_comparisons':PRIOR_STATUS,
       'bootstrap_n':BOOTSTRAP_N,'bootstrap_seed':BOOTSTRAP_SEED,
       'post_selection_bootstrap_caveat':True,
       'train_weeks':'0-24','validation_week':25,'test_data_read':False,
       'final_refit_performed':False,'selection':selection}
    # Pandas Series.to_dict may hold np scalar types; convert via JSON-safe recursion.
    def json_safe(obj):
        if isinstance(obj,dict):return {str(k):json_safe(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)):return [json_safe(v) for v in obj]
        if isinstance(obj,np.integer):return int(obj)
        if isinstance(obj,np.floating):return float(obj)
        if isinstance(obj,np.bool_):return bool(obj)
        return obj
    write_json_atomic(REVIEW/'selection_and_manifest.json',json_safe(MANIFEST_OUT))
    (REVIEW/'README_results.md').write_text(
        '# VK-LSVD FM · validation-only review bundle\n\n'
        'Train weeks 0-24, validation week 25, fixed 3000 users and 2000 items. '
        'Full FM 45 training trajectories, checkpoints 5/10/15 (135 evaluations). '
        'Primary: Eligible NDCG@10 with linear gains 0/1/2. '
        'Ablation uses fixed hyperparameters; seed robustness is descriptive. '
        'Bootstrap intervals do not remove hyperparameter selection bias. '
        'uint64 user_id/item_id columns must NEVER be converted to float. '
        'No test labels or test metrics; no raw data or model checkpoints in bundle.\n',
        encoding='utf-8')
    require((REVIEW/'selection_and_manifest.json').is_file()
        and (REVIEW/'all_135_per_user.parquet').is_file()
        and len(RESULTS)==135,'Не готов review bundle')
    print('PASS. REVIEW BUNDLE:',REVIEW)
    print('Результаты сохранены.')
    print('TEST не читался; final refit не выполнялся.')
