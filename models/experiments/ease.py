"""ease experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from IPython.display import display, Markdown

for name in ("pyarrow", "scipy", "plotly"):
    if importlib.util.find_spec(name) is None:
        raise ModuleNotFoundError(
            f"В текущем Jupyter kernel отсутствует {name}. "
            f"Установи через %pip install {name}, перезапусти kernel и запусти Run All."
        )
import scipy
import plotly
import plotly.express as px

# Явные проверки и контрольные суммы: используются во всех следующих ячейках.
def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path):
    """Побайтовый SHA-256 локального файла, без загрузки его целиком в память."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uint64_ids(values):
    """Алгоритм split audit: sort(uint64 IDs) -> сырые байты -> SHA-256."""
    arr = np.asarray(values)
    require(arr.ndim == 1 and arr.dtype.kind in "iu",
            "ID должны быть одномерным целочисленным массивом")
    require(not (arr < 0).any(), "ID не могут быть отрицательными")
    return hashlib.sha256(np.sort(arr.astype(np.uint64)).tobytes()).hexdigest()


# Ищем корень строго среди текущей папки и её родителей, но НЕ создаём новый проект.
def locate_project_root(start: Path) -> Path:
    for folder in (start, *start.parents):
        if (folder / "data/work/config.json").is_file() and (folder / "data/work/interactions.parquet").is_file():
            return folder
    raise FileNotFoundError(
        "Не найден существующий vk_lsvd_eda_starter с data/work/config.json и data/work/interactions.parquet. "
        f"Рабочая директория Jupyter: {start}. Открой notebook из исходного проекта."
    )

ROOT = locate_project_root(Path.cwd().resolve())
FILES = {
    "config": ROOT / "data/work/config.json",
    "manifest": ROOT / "data/work/manifest.json",
    "revision": ROOT / "data/source/source_revision.json",
    "train": ROOT / "data/work/interactions.parquet",
    "users": ROOT / "data/work/users_metadata.parquet",
    "items": ROOT / "data/work/items_metadata.parquet",
    "validation": ROOT / "data/source/subsamples/up0.001_ip0.001/validation/week_25.parquet",
}
OUT = ROOT / "results" / "ease" / "validation"

# Зафиксированные при предыдущем EDA/evaluator интеграционном тесте fingerprint-и.
EXPECTED_FILE_SHA256 = {
    "train": "0c3a20a9729e30aeb650b28379d34663d3d2ec914bfe0ee3ca0a17d0904790a0",
    "users": "c0769c126e96499364f902362902690774005947d9fe23e097e43718d26d5cb2",
    "items": "3140e6749978a817e552a510c91f0c402eb93a639a096dc81c0bc99526b27d71",
    "validation": "d654b40a7f7e2086054c6131767967f84289fafd2364f566b984cc7d408917f5",
}
EXPECTED_ID_SHA256 = {
    "users": "ce51fd99e116ae5884714fcaf2244b6b99c4d5268ed961426ea86b8277c7780b",
    "items": "e69b0688d7d2ff4a68375a9bc711026d58899b565ce93c4c4c94692c29b59e76",
}
EXPECTED_MODULE_SHA256 = {
    "models/ease.py": "f231fb182fc4f65a749f0576a56e8c5cacc6abae29d3e324eab3e5d98bc90735",
    "models/utils/evaluation/__init__.py": "e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c",
    "models/utils/evaluation/_validation.py": "6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405",
    "models/utils/evaluation/bootstrap.py": "9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221",
    "models/utils/evaluation/metrics.py": "14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938",
    "models/utils/evaluation/pipeline.py": "27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f",
    "models/utils/evaluation/ranking.py": "5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2",
    "models/utils/evaluation/relevance.py": "ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1",
}

# Сетка согласована до проведения эксперимента. Не уточнять lambda автоматически.
SIGNALS = ("binary", "graded")
LAMBDAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
GRID = [(signal, lam) for signal in SIGNALS for lam in LAMBDAS]
TOP_K = 10
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
RUN_FULL_GRID = True
require(len(GRID)==18 and len(set(GRID))==18, "Сетка EASE должна содержать 18 точек")


file_hashes = {name: sha256_file(FILES[name]) for name in EXPECTED_FILE_SHA256}
for name, expected in EXPECTED_FILE_SHA256.items():
    require(file_hashes[name] == expected,
            f"Исходный {name} Parquet ИЗМЕНИЛСЯ: ожидали {expected}, получили {file_hashes[name]}")
module_hashes = {name: sha256_file(ROOT / name) for name in EXPECTED_MODULE_SHA256}
for name, expected in EXPECTED_MODULE_SHA256.items():
    require(module_hashes[name] == expected,
            f"Реализация {name} не совпадает с ожидаемой версией: "
            f"ожидали {expected}, получили {module_hashes[name]}")

CFG = json.loads(FILES["config"].read_text(encoding="utf-8"))
MANIFEST = json.loads(FILES["manifest"].read_text(encoding="utf-8"))
REVISION = json.loads(FILES["revision"].read_text(encoding="utf-8"))
require(CFG["dataset_repo"] == MANIFEST["source"] == REVISION["repo"] == "deepvk/VK-LSVD",
        "Изменён исходный датасет")
require(CFG["subsample"] == MANIFEST["original_slice"] == "up0.001_ip0.001",
        "Не та официальная подвыборка")
require(REVISION["revision"] == MANIFEST["source_revision"] ==
        "084a18386e149a454dd6dbe0123478e75c6b09ce", "Изменена ревизия VK-LSVD")
require(MANIFEST["history_weeks"] == list(range(25)), "Не те недели первоначальной истории")
require(CFG["sample_users"] == MANIFEST["sample_users_requested"] == 3000 and
        CFG["sample_items"] == MANIFEST["sample_items_requested"] == 2000,
        "Изменён размер исходной ID-выборки")
require(CFG["random_seed"] == MANIFEST["seed_users"] == 42 and
        MANIFEST["seed_items"] == 43, "Сменились seeds исторической выборки")
require(MANIFEST["interactions_rows"] == 1454216, "Не тот размер исходного train")


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.ease import EASE, prepare_ease_data
from models.utils.evaluation import label_relevance, evaluate_scores, bootstrap_mean, bootstrap_paired_difference
print("PASS: исходные Parquet, manifest, EASE.py и evaluator совпадают с принятыми версиями")


TRAIN_COLS = ["user_id", "item_id", "week", "timespent", "like", "bookmark", "dislike"]
VAL_COLS = ["user_id", "item_id", "timespent", "like", "bookmark", "dislike"]
train_raw = pd.read_parquet(FILES["train"], columns=TRAIN_COLS)
users_meta = pd.read_parquet(FILES["users"], columns=["user_id"])
items_meta = pd.read_parquet(FILES["items"], columns=["item_id", "duration"])
val_raw = pd.read_parquet(FILES["validation"], columns=VAL_COLS)

for name, frame, id_cols in (
    ("train", train_raw, ["user_id", "item_id"]),
    ("users_metadata", users_meta, ["user_id"]),
    ("items_metadata", items_meta, ["item_id"]),
    ("validation_source", val_raw, ["user_id", "item_id"]),
):
    require(not frame.empty, f"{name} пустая таблица")
    require(frame[id_cols].notna().all().all(), f"{name}: пропущенные ID")
    for field in id_cols:
        require(frame[field].dtype.kind in "iu", f"{name}.{field}: ID не целочисленные")
    require(not frame.duplicated(id_cols).any(), f"{name}: дубли {id_cols}")

require(len(train_raw) == 1454216, "Число train-событий изменилось")
require(set(train_raw.week.unique()) == set(range(25)), "Неправильные недели train")
require(len(users_meta) == 3000 and len(items_meta) == 2000, "Неправильные размеры ID-каталога")
require(len(val_raw) == 519839, "Validation-source отличается от официальной недели 25")
USER_IDS = np.sort(users_meta.user_id.to_numpy(dtype=np.uint64))
ITEM_IDS = np.sort(items_meta.item_id.to_numpy(dtype=np.uint64))
user_hash = sha256_uint64_ids(USER_IDS)
item_hash = sha256_uint64_ids(ITEM_IDS)
require(user_hash == EXPECTED_ID_SHA256["users"], "ДРУГИЕ ПОЛЬЗОВАТЕЛИ: user-ID hash не совпал")
require(item_hash == EXPECTED_ID_SHA256["items"], "ДРУГИЕ ВИДЕО: item-ID hash не совпал")
require(set(train_raw.user_id) == set(USER_IDS) and set(train_raw.item_id) == set(ITEM_IDS),
        "Train не соответствует исходному EDA-набору ID")

validation_raw = val_raw.loc[
    val_raw.user_id.isin(USER_IDS) & val_raw.item_id.isin(ITEM_IDS)
].copy()
validation_raw["week"] = np.int16(25)  # Гарантировано именем и хешем week_25.parquet.
require(len(validation_raw) == 16158, "Validation-фильтрация не дала 16 158 событий")
require(validation_raw.user_id.nunique() == 2819 and validation_raw.item_id.nunique() == 839,
        "Число validation-пользователей/объектов не совпало с аудитом")
require(not validation_raw.duplicated(["user_id", "item_id"]).any(), "Дубли пар в validation")
require(not pd.MultiIndex.from_frame(validation_raw[["user_id", "item_id"]]).isin(
    pd.MultiIndex.from_frame(train_raw[["user_id", "item_id"]])).any(),
    "Есть пересечение train и validation user–item пар")

seen = train_raw.groupby("user_id")["item_id"].nunique().reindex(USER_IDS, fill_value=0)
unseen = len(ITEM_IDS) - seen
require(int(unseen.min()) == 1124 and float(unseen.median()) == 1512,
        "Изменилось число доступных кандидатов: min=1124, median=1512")
require((unseen >= 10).all(), "Есть пользователь с менее чем 10 кандидатами")

print("PASS: именно та же EDA-выборка, без новых пользователей или видео")
print("Train:", len(train_raw), "| users:", len(USER_IDS), "| videos:", len(ITEM_IDS))
print("Validation:", len(validation_raw), "из", len(val_raw), "исходных событий")
print("user ID SHA-256:", user_hash)
print("item ID SHA-256:", item_hash)
print("Доступных кандидатов на пользователя: min", int(unseen.min()),
      ", median", int(unseen.median()))
del val_raw, users_meta


require(items_meta.duration.notna().all() and items_meta.duration.gt(0).all(),
        "Некорректные durations в EDA items_metadata")
item_duration = items_meta[["item_id", "duration"]]
train_joined = train_raw.merge(item_duration, on="item_id", how="left", validate="many_to_one", sort=False)
val_joined = validation_raw.merge(item_duration, on="item_id", how="left", validate="many_to_one", sort=False)
require(len(train_joined) == 1454216 and len(val_joined) == 16158,
        "Соединение с duration изменило число наблюдений")
require(train_joined.duration.notna().all() and val_joined.duration.notna().all(),
        "Для части событий не найдена duration")
train_labeled = label_relevance(train_joined)
val_labeled = label_relevance(val_joined)

train_levels = train_labeled.rel.value_counts().reindex([0, 1, 2], fill_value=0)
val_levels = val_labeled.rel.value_counts().reindex([0, 1, 2], fill_value=0)
require(np.array_equal(train_levels.to_numpy(), [922110, 148820, 383286]),
        "Train-rel отличается от утверждённой EDA-шкалы")
require(np.array_equal(val_levels.to_numpy(), [11352, 1462, 3344]),
        "Validation-rel отличается от аудита")
require(int((val_labeled.rel >= 1).sum()) == 4806, "Validation positives отличаются от 4 806")
require(val_labeled.loc[val_labeled.rel >= 1, "user_id"].nunique() == 2082,
        "Eligible-пользователей должно быть 2 082")

# Отдаём обеим моделям одинаковый размеченный train; EASE внутри создаёт Binary/Graded матрицу X.
HISTORY = train_labeled[["user_id", "item_id", "week", "rel"]].copy()
HOLDOUT = val_labeled[["user_id", "item_id", "week", "rel"]].copy()
require(HISTORY.week.between(0, 24).all() and HOLDOUT.week.eq(25).all(),
        "Неверная временная принадлежность размеченных событий")
require(len(HISTORY) == 1454216 and len(HOLDOUT) == 16158, "Неправильный размер размеченных таблиц")

label_table = pd.DataFrame({"rel": [0, 1, 2],
                            "train": train_levels.to_numpy(dtype=np.int64),
                            "validation": val_levels.to_numpy(dtype=np.int64)})
display(label_table)
print("PASS: 922110 / 148820 / 383286 (train); 11352 / 1462 / 3344 (val)")
print("Validation eligible: 2082 / 3000; ineligible: 918 / 3000")
del train_raw, validation_raw, train_joined, val_joined, train_labeled, val_labeled, items_meta
# release extra references to large frames before fitting
_ = gc.collect()


tiny_u = np.array([2**63+5, 2**63+6], dtype=np.uint64)
tiny_i = np.arange(10, 26, dtype=np.uint64)
tiny_h = pd.DataFrame({'user_id': np.array([tiny_u[0],tiny_u[0],tiny_u[1]], dtype='uint64'),
                       'item_id': np.array([tiny_i[0],tiny_i[1],tiny_i[1]], dtype='uint64'),
                       'week':[0,0,0], 'rel':[0,2,1]})
tiny_v = pd.DataFrame({'user_id': np.array([tiny_u[0],tiny_u[1]], dtype='uint64'),
                       'item_id': np.array([tiny_i[2],tiny_i[3]], dtype='uint64'),
                       'week':[1,1], 'rel':[2,1]})
for mode in SIGNALS:
    p=prepare_ease_data(tiny_h,tiny_u,tiny_i,signal=mode,history_start_week=0,history_end_week=0)
    model=EASE(10.,mode).fit_prepared(p)
    scores=model.score()
    require(scores.shape==(2,16) and scores.dtype==np.float64 and np.isfinite(scores).all(),
            'Неправильные EASE scores')
    require(np.array_equal(np.diag(model.coef_),np.zeros(16)), 'Ненулевая диагональ EASE')
    ev=evaluate_scores(scores,tiny_u,tiny_i,tiny_h,tiny_v,k=10,
                       history_start_week=0,history_end_week=0,holdout_week=1)
    require(len(ev.recommendations)==20 and ev.summary['n_eligible']==2,'EASE/evaluator smoke test')
    require(not pd.MultiIndex.from_frame(ev.recommendations[['user_id','item_id']]).isin(
                    pd.MultiIndex.from_frame(tiny_h[['user_id','item_id']])).any(),
            'Оценщик пропустил известное видео (в том числе rel=0)')
    print(mode,'PASS: EASE → evaluator → 10 новых видео на пользователя')


PREPARED={}
prep_times={}
for signal in SIGNALS:
    start=time.perf_counter()
    data=prepare_ease_data(HISTORY,USER_IDS,ITEM_IDS,signal=signal,
                           history_start_week=0,history_end_week=24)
    prep_times[signal]=float(time.perf_counter()-start)
    require(data.X.shape==(3000,2000) and data.G.shape==(2000,2000),
            'Подготовка EASE нарушила исходные размеры')
    require(np.isfinite(data.G).all(), 'Неконечные значения Gram')
    PREPARED[signal]=data
    print(f'{signal:6s}: подготовка {prep_times[signal]:.2f} с; '
          f'положительных/ненулевых ячеек {np.count_nonzero(data.X):,}')


from itertools import product
import os

CACHE = OUT / 'checkpoint_cache'
REVIEW = OUT / 'review_bundle'
CACHE.mkdir(parents=True,exist_ok=True)

run_identity = {
  'experiment': 'EASE_validation_v1', 'dataset': 'deepvk/VK-LSVD',
  'source_revision': REVISION['revision'],
  'input_sha256': file_hashes,'module_sha256': module_hashes,
  'user_ids_sha256':user_hash,'item_ids_sha256':item_hash,
  'train_weeks':[0,24],'validation_week':25,
  'signals':list(SIGNALS),'lambdas':list(LAMBDAS),'k':TOP_K,
  'n_train':1454216,'n_validation':16158,'eligible':2082,
  'tie_break':'highest Eligible NDCG, then binary, then smaller lambda',
}
identity_json=json.dumps(run_identity,sort_keys=True,ensure_ascii=False,separators=(',',':'))
IDENTITY_SHA=hashlib.sha256(identity_json.encode('utf-8')).hexdigest()
identity_path=OUT/'run_identity.json'
if identity_path.exists():
    old=json.loads(identity_path.read_text(encoding='utf-8'))
    require(old==run_identity,'В каталоге есть кеш ОТ ДРУГОГО эксперимента! '
            'Не смешивай результаты: переименуй results/ease/validation и повтори запуск.')
else:
    require(not list(CACHE.glob('*')),'Неизвестный кеш без run_identity — небезопасно продолжать')
    identity_path.write_text(json.dumps(run_identity,indent=2,ensure_ascii=False),encoding='utf-8')

def config_stem(signal,lam):
    require(signal in SIGNALS and lam in LAMBDAS,'Неизвестная конфигурация')
    return f'{signal}_lambda_{int(lam)}'

def files_for(signal,lam):
    stem=config_stem(signal,lam)
    return (CACHE/f'{stem}_per_user.parquet',CACHE/f'{stem}_recommendations.parquet',
            CACHE/f'{stem}_complete.json')

def save_parquet_atomic(df, dest):
    tmp=dest.with_name(dest.name+'.tmp')
    df.to_parquet(tmp,index=False)
    os.replace(tmp,dest)

all_result_rows=[]
if RUN_FULL_GRID:
  for signal,lam in GRID:
    per_path,rec_path,complete_path=files_for(signal,lam)
    stem=config_stem(signal,lam)
    if complete_path.exists():
        d=json.loads(complete_path.read_text(encoding='utf-8'))
        require(d.get('identity_sha256')==IDENTITY_SHA and
                d.get('signal')==signal and d.get('regularization')==lam and
                per_path.is_file() and rec_path.is_file(),
                f'{stem}: кеш не соответствует эксперименту')
        require(sha256_file(per_path)==d.get('per_user_sha256') and
                sha256_file(rec_path)==d.get('recommendations_sha256'),
                f'{stem}: файл кеша повреждён')
        all_result_rows.append(d['row'])
        print(f'↪ {stem}: готово (кеш), NDCG={d["row"]["eligible_ndcg_at_10"]:.6f}',flush=True)
        continue
    if any(x.exists() for x in (per_path,rec_path)):
        print(f'↺ {stem}: неполный запуск, безопасно пересчитываем.',flush=True)
    t0=time.perf_counter()
    model=EASE(regularization=lam,signal=signal).fit_prepared(PREPARED[signal])
    fit_seconds=float(time.perf_counter()-t0)
    B=model.coef_
    require(B.shape==(2000,2000) and np.diag(B).tolist()==[0.0]*2000,
            f'{stem}: неправильная матрица коэффициентов')
    t1=time.perf_counter()
    scores=model.score()
    score_seconds=float(time.perf_counter()-t1)
    require(scores.shape==(3000,2000) and np.isfinite(scores).all(),
            f'{stem}: некорректные scores')
    t2=time.perf_counter()
    ev=evaluate_scores(scores,USER_IDS,ITEM_IDS,HISTORY,HOLDOUT,k=TOP_K,
                       history_start_week=0,history_end_week=24,holdout_week=25)
    evaluation_seconds=float(time.perf_counter()-t2)
    summary=ev.summary
    require(summary['n_users']==3000 and summary['n_eligible']==2082 and
            summary['n_history']==1454216 and summary['n_holdout']==16158 and
            summary['holdout_week_checked'] is True and
            len(ev.per_user)==3000 and len(ev.recommendations)==30000,
            f'{stem}: evaluator нарушил фиксированный протокол')
    metrics={
      'eligible_ndcg_at_10':float(summary['eligible']['ndcg_at_10']),
      'eligible_recall_at_10':float(summary['eligible']['recall_at_10']),
      'eligible_map_at_10':float(summary['eligible']['map_at_10']),
      'all_users_ndcg_at_10':float(summary['all_users']['ndcg_at_10']),
      'all_users_recall_at_10':float(summary['all_users']['recall_at_10']),
      'all_users_map_at_10':float(summary['all_users']['map_at_10']),
      'eligible_share':float(summary['eligible_share']),
    }
    require(all(np.isfinite(v) and 0<=v<=1+1e-12 for v in metrics.values()),
            f'{stem}: недопустимое значение метрик')
    require(abs(metrics['eligible_share']-2082/3000)<1e-12,
            f'{stem}: доля eligible изменилась')
    require(np.array_equal(ev.per_user.user_id.to_numpy(dtype=np.uint64),USER_IDS),
            f'{stem}: другой порядок пользователей')
    require(int(ev.per_user.eligible.sum())==2082,f'{stem}: иная eligibility-маска')
    require(not pd.MultiIndex.from_frame(ev.recommendations[['user_id','item_id']]).isin(
                     pd.MultiIndex.from_frame(HISTORY[['user_id','item_id']])).any(),
            f'{stem}: просмотры попали в выдачу')
    require((ev.recommendations.groupby('user_id',sort=False).item_id.nunique()==10).all(),
            f'{stem}: Top-10 содержит повторы')
    offdiag=~np.eye(2000,dtype=bool)
    diag={
      'n_negative_coefficients':int(np.count_nonzero(B[offdiag]<0)),
      'n_positive_coefficients':int(np.count_nonzero(B[offdiag]>0)),
      'fraction_negative_coefficients':float(np.mean(B[offdiag]<0)),
      'mean_abs_coefficient':float(np.mean(np.abs(B[offdiag]))),
      'max_abs_coefficient':float(np.max(np.abs(B[offdiag]))),
      'n_negative_scores':int(np.count_nonzero(scores<0)),
      'unique_recommended_items':int(ev.recommendations.item_id.nunique()),
      'train_objective':float(0.5*np.sum((PREPARED[signal].X-scores)**2)
                              +0.5*lam*np.sum(B**2)),
    }
    row={'signal':signal,'regularization':float(lam),**metrics,**diag,
         'prepare_seconds':prep_times[signal],'fit_seconds':fit_seconds,
         'score_seconds':score_seconds,'evaluation_seconds':evaluation_seconds,
         'n_users':3000,'n_eligible':2082,'n_items':2000,'n_history':1454216}
    save_parquet_atomic(ev.per_user,per_path)
    save_parquet_atomic(ev.recommendations,rec_path)
    record={'identity_sha256':IDENTITY_SHA,'signal':signal,'regularization':float(lam),
            'per_user_sha256':sha256_file(per_path),
            'recommendations_sha256':sha256_file(rec_path),'row':row}
    tmp=complete_path.with_name(complete_path.name+'.tmp')
    tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,complete_path)  # завершающий маркер пишем ПОСЛЕ обоих Parquet
    all_result_rows.append(row)
    print(f'✓ {stem:25s}: NDCG={metrics["eligible_ndcg_at_10"]:.6f}, '
          f'Recall={metrics["eligible_recall_at_10"]:.6f}, '
          f'MAP={metrics["eligible_map_at_10"]:.6f}, '
          f'fit={fit_seconds:.2f}s, eval={evaluation_seconds:.2f}s',flush=True)
    del model,B,scores,ev
    gc.collect()
  print('Все 18 комбинаций рассчитаны или восстановлены из проверенного кеша.')
else:
  print('RUN_FULL_GRID=False: итоговый выбор не проводится; включи True для завершения.')


if RUN_FULL_GRID:
    RESULTS=pd.DataFrame(all_result_rows)
    require(len(RESULTS)==18 and not RESULTS.duplicated(['signal','regularization']).any()
            and set(zip(RESULTS.signal,RESULTS.regularization))==set(GRID),
            'Не все 18 уникальных EASE конфигураций завершены')
    require((RESULTS.n_users==3000).all() and (RESULTS.n_eligible==2082).all(),
            'Разные размеры eligible population')
    RESULTS['signal_order']=RESULTS.signal.map({'binary':0,'graded':1}).astype(int)
    BEST_ORDER=RESULTS.sort_values(['eligible_ndcg_at_10','signal_order','regularization'],
                                  ascending=[False,True,True],kind='mergesort')
    BEST_OVERALL=BEST_ORDER.iloc[0]
    BEST_PER_MODE=(BEST_ORDER.drop_duplicates('signal')
                   .sort_values('signal_order',kind='mergesort'))
    # DataFrame.style требует необязательную зависимость jinja2.
    # Обычный DataFrame отображается в Jupyter без Styler/jinja2.
    visible_columns = ['signal', 'regularization', 'eligible_ndcg_at_10',
                       'eligible_recall_at_10', 'eligible_map_at_10',
                       'fit_seconds', 'unique_recommended_items']
    results_table = (RESULTS[visible_columns]
                     .sort_values(['signal', 'regularization'])
                     .round({'eligible_ndcg_at_10': 6,
                             'eligible_recall_at_10': 6,
                             'eligible_map_at_10': 6,
                             'fit_seconds': 2}))
    display(results_table)
    print('BEST OVERALL:',BEST_OVERALL[['signal','regularization','eligible_ndcg_at_10',
          'eligible_recall_at_10','eligible_map_at_10']].to_dict())
    print('Лучшие по каждому signal:')
    display(BEST_PER_MODE[['signal','regularization','eligible_ndcg_at_10',
                          'eligible_recall_at_10','eligible_map_at_10']])


if RUN_FULL_GRID:
    figures={}
    specs=[('eligible_ndcg_at_10','NDCG@10'),('eligible_recall_at_10','Recall@10'),
           ('eligible_map_at_10','MAP@10')]
    for metric,label in specs:
        fig=px.line(RESULTS.sort_values('regularization'),x='regularization',y=metric,
                    color='signal',markers=True,log_x=True,
                    title=f'EASE: Eligible {label} vs L2 regularization λ',
                    labels={'regularization':'λ (логарифмическая шкала)','signal':'Train signal',metric:label},
                    hover_data=['fit_seconds','unique_recommended_items','fraction_negative_coefficients'])
        if metric=='eligible_ndcg_at_10':
            fig.add_hline(y=0.020702,line_dash='dot',line_color='gray',
                          annotation_text='iALS 0.020702',annotation_position='top left')
            fig.add_hline(y=0.018522,line_dash='dash',line_color='gray',
                          annotation_text='ItemKNN 0.018522',annotation_position='bottom left')
        fig.update_layout(template='plotly_white',height=430)
        figures[metric]=fig
        fig.show()
    diag_fig=px.line(RESULTS.sort_values('regularization'),x='regularization',
                     y='fraction_negative_coefficients',color='signal',markers=True,log_x=True,
                     title='Доля отрицательных коэффициентов EASE по λ',
                     labels={'regularization':'λ','fraction_negative_coefficients':'Доля отрицательных B'})
    diag_fig.update_layout(template='plotly_white',height=400)
    figures['fraction_negative_coefficients']=diag_fig
    diag_fig.show()
    times_fig=px.bar(RESULTS.sort_values('regularization'),x='regularization',y='fit_seconds',
                     color='signal',barmode='group',log_x=True,
                     title='Фактическое время fit на текущем компьютере',
                     labels={'regularization':'λ','fit_seconds':'Секунды'})
    times_fig.update_layout(template='plotly_white',height=390)
    figures['fit_seconds']=times_fig
    times_fig.show()


if RUN_FULL_GRID:
    winner_per_user={}
    winner_recs={}
    for _,r in BEST_PER_MODE.iterrows():
        mode=str(r.signal)
        pu_path, rec_path, _=files_for(mode,float(r.regularization))
        pu=pd.read_parquet(pu_path)
        rec=pd.read_parquet(rec_path)
        require(len(pu)==3000 and len(rec)==30000 and
                np.array_equal(pu.user_id.to_numpy(dtype='uint64'),USER_IDS),
                f'Некорректный кеш победителя {mode}')
        winner_per_user[mode]=pu
        winner_recs[mode]=rec
    interval_rows=[]
    for mode,pu in winner_per_user.items():
        for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
            for group in ('eligible','all_users'):
                out=bootstrap_mean(pu,metric=metric,group=group,n_bootstrap=BOOTSTRAP_N,
                                   seed=BOOTSTRAP_SEED,ci=.95)
                interval_rows.append({'signal':mode,'metric':metric,'group':group,**out})
    INTERVALS=pd.DataFrame(interval_rows)
    paired_rows=[]
    for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
        for group in ('eligible','all_users'):
            out=bootstrap_paired_difference(winner_per_user['graded'],winner_per_user['binary'],
                    metric=metric,group=group,n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED,ci=.95)
            paired_rows.append({'comparison':'EASE graded - EASE binary', 'metric':metric,'group':group,**out})
    # Дополнительные pairwise сравнения: только по РЕАЛЬНЫМ файлам, если они доступны.
    prior_paths={
      'ItemKNN binary normalized':ROOT/'results/item_knn/validation/winner_binary_normalized_per_user.csv',
      'iALS binary lowrank':ROOT/'results/ials/lowrank_validation/review_bundle/winner_binary_per_user.csv',
    }
    previous_status={}
    for label,path in prior_paths.items():
        if not path.is_file():
            previous_status[label]='отсутствует файл: сравнение пропущено'
            continue
        prior=pd.read_csv(path,dtype={'user_id':'uint64'})
        require(len(prior)==3000 and not prior.user_id.duplicated().any(),
                f'{label}: прежний файл не соответствует 3000 исходным пользователям')
        require(np.array_equal(np.sort(prior.user_id.to_numpy(dtype='uint64')), USER_IDS),
                f'{label}: другой набор ID')
        for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
            for group in ('eligible','all_users'):
                out=bootstrap_paired_difference(winner_per_user[str(BEST_OVERALL.signal)],prior,
                    metric=metric,group=group,n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED,ci=.95)
                paired_rows.append({'comparison':f'EASE best - {label}','metric':metric,'group':group,**out})
        previous_status[label]='парный bootstrap рассчитан'
    PAIRED=pd.DataFrame(paired_rows)
    print('Bootstrap 95% CI (post-selection, exploratory):')
    display(INTERVALS.loc[INTERVALS.group=='eligible',
         ['signal','metric','estimate','ci_low','ci_high']])
    print('Парные интервалы, Eligible:')
    display(PAIRED.loc[PAIRED.group=='eligible',
         ['comparison','metric','estimate','ci_low','ci_high']])
    print('Прошлые модели:',previous_status)


if RUN_FULL_GRID:
    REVIEW.mkdir(parents=True,exist_ok=True)
    RESULTS.drop(columns='signal_order').sort_values(['signal','regularization']).to_csv(
        REVIEW/'all_18_validation_metrics.csv',index=False)
    BEST_PER_MODE.drop(columns='signal_order').to_csv(
        REVIEW/'best_binary_and_graded.csv',index=False)
    INTERVALS.to_csv(REVIEW/'bootstrap_best_modes.csv',index=False)
    PAIRED.to_csv(REVIEW/'paired_differences.csv',index=False)
    label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
    per_user_all=[]
    for signal,lam in GRID:
        pu_path,_,_=files_for(signal,lam)
        pu=pd.read_parquet(pu_path)
        require(len(pu)==3000 and int(pu.eligible.sum())==2082,
                f'Неверный per-user cache {signal} {lam}')
        pu.insert(0,'signal',signal)
        pu.insert(1,'regularization',float(lam))
        per_user_all.append(pu)
    ALL_PER_USER=pd.concat(per_user_all,ignore_index=True)
    require(len(ALL_PER_USER)==54000,'Не 18 × 3000 пользователей в per-user')
    ALL_PER_USER.to_csv(REVIEW/'all_18_per_user.csv',index=False)
    for mode in SIGNALS:
        winner_per_user[mode].to_csv(REVIEW/f'winner_{mode}_per_user.csv',index=False)
        winner_recs[mode].to_csv(REVIEW/f'winner_{mode}_recommendations.csv',index=False)
    for name,fig in figures.items():
        fig.write_html(str(REVIEW/f'ease_{name}.html'),include_plotlyjs='cdn',full_html=True)
    selection={
      'metric':'Eligible NDCG@10, linear gains for 0/1/2',
      'tie_break':'Higher NDCG, then binary, then smaller lambda',
      'best_overall':{'signal':str(BEST_OVERALL.signal),
                      'regularization':float(BEST_OVERALL.regularization),
                      'eligible_ndcg_at_10':float(BEST_OVERALL.eligible_ndcg_at_10),
                      'eligible_recall_at_10':float(BEST_OVERALL.eligible_recall_at_10),
                      'eligible_map_at_10':float(BEST_OVERALL.eligible_map_at_10)},
      'best_per_signal':BEST_PER_MODE[['signal','regularization',
          'eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']].to_dict(orient='records')
    }
    manifest={
      'status':'PASS_VALIDATION_ONLY','experiment':'EASE main grid v1',
      'source_revision':REVISION['revision'],'file_sha256':file_hashes,
      'module_sha256':module_hashes,'user_id_sha256':user_hash,'item_id_sha256':item_hash,
      'run_identity_sha256':IDENTITY_SHA,'n_train':1454216,'n_val':16158,
      'n_users':3000,'n_items':2000,'eligible':2082,
      'train_weeks':'0–24','validation_week':25,
      'n_configurations':18,'n_all_per_user_rows':len(ALL_PER_USER),
      'signals':list(SIGNALS),'lambdas':list(LAMBDAS),
      'bootstrap_n':BOOTSTRAP_N,'bootstrap_seed':BOOTSTRAP_SEED,
      'previous_model_comparison':previous_status,'selection':selection,
      'test_data_read':False,'test_metrics_computed':False,
      'future_refit_performed':False,'bootstrap_post_selection_warning':True,
    }
    (REVIEW/'selection_and_manifest.json').write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (REVIEW/'README_results.md').write_text(
        '# VK-LSVD — EASE validation\n\n'
        'Train 0–24; validation week 25; fixed 3000 users and 2000 videos. '
        '18 configurations (2 signals × 9 lambda); test week 26 untouched. '
        'Primary: Eligible NDCG@10, linear gains 0/1/2. '
        'Bootstrap intervals are descriptive and can be selection-biased. '
        'All saved per-user data have unsigned 64-bit IDs; read with uint64, never float. '
        'Source data, training coefficients and checkpoint_cache are not included.\n',
        encoding='utf-8')
    require(len(list(REVIEW.glob('winner_*_per_user.csv')))==2 and
            len(list(REVIEW.glob('winner_*_recommendations.csv')))==2,
            'Не все победители сохранены')
    print('PASS: review_bundle готов:',REVIEW)
    print('Результаты сохранены.')
    print('TEST week_26 не загружался и не оценивался.')
