"""ials lowrank experiment on train/validation."""
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
OUT = ROOT / "results" / "ials" / "lowrank_validation"
PREV_OUT = ROOT / "results" / "ials" / "validation"
PREV_REVIEW = PREV_OUT / "review_bundle"
PREV_CACHE = PREV_OUT / "checkpoint_cache"

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
    "models/ials.py": "f03a19c7651223a2812d87f3728a2b98a717e635b6fc5b276bf82d94f548c4f3",
    "models/utils/evaluation/__init__.py": "e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c",
    "models/utils/evaluation/_validation.py": "6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405",
    "models/utils/evaluation/bootstrap.py": "9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221",
    "models/utils/evaluation/metrics.py": "14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938",
    "models/utils/evaluation/pipeline.py": "27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f",
    "models/utils/evaluation/ranking.py": "5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2",
    "models/utils/evaluation/relevance.py": "ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1",
}

FACTORS=(2, 4, 6, 8, 10, 12, 14, 16)
CONFIDENCE_MODES=("binary", "graded")
MODE_PARAMS = {
    "binary": {"alpha": 40.0, "regularization": 10.0},
    "graded": {"alpha": 10.0, "regularization": 1.0},
}
CHECKPOINTS=(5, 10, 15)
SEED=42
TOP_K=10
BOOTSTRAP_N=2000
BOOTSTRAP_SEED=42
EXPECTED_TRAJECTORIES=16
EXPECTED_CHECKPOINT_EVALUATIONS=48


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
from models import IALS
from models.utils.evaluation import label_relevance, evaluate_scores, bootstrap_mean, bootstrap_paired_difference
require(len(FACTORS)*len(CONFIDENCE_MODES)==EXPECTED_TRAJECTORIES,
        "Должно быть ровно 16 конфигураций low-rank эксперимента")
require(CHECKPOINTS == (5,10,15), "Checkpoints неожиданно изменились")
print("PASS: исходные Parquet, manifest, iALS.py и evaluator совпадают с принятыми версиями")


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

# Отдаём обеим моделям одинаковый размеченный train; IALS внутри преобразует `rel` в preference/confidence.
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


tiny_u = np.array([2**63 + 5, 2**63 + 6], dtype=np.uint64)
tiny_i = np.arange(10, 26, dtype=np.uint64)
tiny_h = pd.DataFrame({
    "user_id": np.array([tiny_u[0], tiny_u[0], tiny_u[1]], dtype=np.uint64),
    "item_id": np.array([tiny_i[0], tiny_i[1], tiny_i[1]], dtype=np.uint64),
    "week": [0,0,0], "rel": [0,2,1]
})
tiny_hold = pd.DataFrame({
    "user_id": np.array([tiny_u[0], tiny_u[1]], dtype=np.uint64),
    "item_id": np.array([tiny_i[2], tiny_i[3]], dtype=np.uint64),
    "week": [1,1], "rel": [2,1]
})
for mode in CONFIDENCE_MODES:
    tiny = IALS(factors=2, alpha=5., regularization=1., confidence_mode=mode)
    tiny.fit(tiny_h, tiny_u, tiny_i, n_iters=1, history_start_week=0, history_end_week=0)
    require(tiny.score().shape == (2,16) and np.isfinite(tiny.score()).all(), "Неверная матрица scores")
    tiny.continue_fit(1)
    tiny_ev = evaluate_scores(tiny.score(), tiny_u, tiny_i, tiny_h, tiny_hold,
                              k=10, history_start_week=0, history_end_week=0, holdout_week=1)
    require(len(tiny_ev.recommendations)==20 and tiny_ev.summary['n_eligible']==2,
            "IALS/evaluator интеграция не прошла")
    require(not pd.MultiIndex.from_frame(tiny_ev.recommendations[['user_id','item_id']]).isin(
        pd.MultiIndex.from_frame(tiny_h[['user_id','item_id']])).any(),
        "В выдачу попали ранее просмотренные объекты")
print("PASS: synthetic binary/graded IALS → 10 recommendations/user → evaluator")


from itertools import product

require(PREV_OUT.exists(),
        'Не найдена прежняя results/ials/validation. Восстанови её из результатов первого эксперимента.')
old_id_path=PREV_OUT/'run_identity.json'
old_csv_path=PREV_REVIEW/'all_72_validation_metrics.csv'
require(old_id_path.is_file() and old_csv_path.is_file(),
        'Нет run_identity.json / review_bundle/all_72_validation_metrics.csv первого iALS-эксперимента')
old_identity=json.loads(old_id_path.read_text(encoding='utf-8'))
require(old_identity['files']==file_hashes and old_identity['modules']==module_hashes
        and old_identity['ids']=={'users':user_hash,'items':item_hash}
        and old_identity['revision']==REVISION['revision']
        and old_identity['seed']==SEED and old_identity['holdout_week']==25
        and old_identity['top_k']==TOP_K,
        'Прежний iALS относится к ДРУГИМ данным, версиям или seed')
require(old_identity['factors']==[16,32,64]
        and old_identity['alphas']==[10.,40.]
        and old_identity['regularizations']==[1.,10.]
        and old_identity['confidence_modes']==['binary','graded']
        and old_identity['checkpoints']==[5,10,15],
        'Прежний iALS был запущен не по утверждённой полной сетке')
OLD_RESULTS=pd.read_csv(old_csv_path)
DIMS=['factors','alpha','regularization','confidence_mode','iteration']
METRICS=['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10',
         'all_users_ndcg_at_10','all_users_recall_at_10','all_users_map_at_10']
require(len(OLD_RESULTS)==72 and not OLD_RESULTS.duplicated(DIMS).any(),
        'Предыдущий результат должен содержать ровно 72 уникальных checkpoints')
require(OLD_RESULTS[METRICS].notna().all().all() and
        (OLD_RESULTS.n_users==3000).all() and (OLD_RESULTS.n_eligible==2082).all(),
        'Неверные метрики/число eligible в старом эксперименте')

# Файлы из прошлого кеша можно переиспользовать ТОЛЬКО если вся траектория f=16 целая.
def stem_for(f, mode):
    p=MODE_PARAMS[mode]
    return f"f{f}_alpha{int(p['alpha'])}_lambda{int(p['regularization'])}_{mode}"

def cached_parts(folder, stem):
    return [folder/f'{stem}_metrics.csv', folder/f'{stem}_training_log.csv'] + [
        folder/f'{stem}_it{it}_{kind}.parquet'
        for it in CHECKPOINTS for kind in ('per_user','recommendations')
    ]

REUSE_F16={}
for mode in CONFIDENCE_MODES:
    stem=stem_for(16,mode)
    old_files=cached_parts(PREV_CACHE,stem)
    can_reuse=all(p.is_file() for p in old_files)
    if can_reuse:
        old_checkpoint_metrics=pd.read_csv(PREV_CACHE/f'{stem}_metrics.csv')
        expected_rows=OLD_RESULTS.loc[
            (OLD_RESULTS.factors==16)&
            (OLD_RESULTS.confidence_mode==mode)&
            (OLD_RESULTS.alpha==MODE_PARAMS[mode]['alpha'])&
            (OLD_RESULTS.regularization==MODE_PARAMS[mode]['regularization'])
        ].sort_values('iteration')
        can_reuse=(len(old_checkpoint_metrics)==3 and len(expected_rows)==3
                   and (old_checkpoint_metrics.sort_values('iteration').iteration.to_numpy()==np.array(CHECKPOINTS)).all()
                   and np.allclose(old_checkpoint_metrics.sort_values('iteration')[METRICS].to_numpy(),
                                   expected_rows[METRICS].to_numpy(),rtol=1e-10,atol=1e-12))
    REUSE_F16[mode]=bool(can_reuse)
    print(f'Previous f=16 / {mode}:', 'reuse validated checkpoints' if can_reuse else 'train again in separate cache')
print('PASS: исходные 72 метрики и их provenance проверены')


CACHE=OUT/'checkpoint_cache'
REVIEW=OUT/'review_bundle'
OUT.mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
GRID=list(product(FACTORS,CONFIDENCE_MODES))
require(len(GRID)==16 and len(set(GRID))==16, 'Некорректная сетка')
run_identity={
    'experiment':'iALS low-rank extension, fixed previous per-mode alpha and lambda',
    'dataset':'deepvk/VK-LSVD', 'revision':REVISION['revision'],
    'input_files_sha256':file_hashes, 'model_and_evaluator_sha256':module_hashes,
    'ids_sha256':{'users':user_hash,'items':item_hash},
    'prior_run_identity_sha256':sha256_file(old_id_path),
    'prior_all_72_metrics_sha256':sha256_file(old_csv_path),
    'factors':list(FACTORS),'confidence_modes':list(CONFIDENCE_MODES),
    'mode_params':MODE_PARAMS,'checkpoints':list(CHECKPOINTS),
    'random_seed':SEED,'top_k':TOP_K,'validation_week':25,
}
identity_file=OUT/'run_identity.json'
if identity_file.exists():
    require(json.loads(identity_file.read_text(encoding='utf-8'))==run_identity,
            'Найден кеш другого low-rank эксперимента. Не подменяй SHA: перемести всю папку '
            'results/ials/lowrank_validation отдельно и запусти заново.')
else:
    require(not any(CACHE.iterdir()), 'Найден неопознанный кеш без run_identity.json')
    identity_file.write_text(json.dumps(run_identity,indent=2,ensure_ascii=False),encoding='utf-8')

# Однозначно выбираем источник персональных метрик и рекомендаций.
def checkpoint_dir(f, mode):
    return PREV_CACHE if f==16 and REUSE_F16[mode] else CACHE

def checkpoint_path(f, mode, it, kind):
    return checkpoint_dir(f,mode)/f'{stem_for(f,mode)}_it{it}_{kind}.parquet'

print('Всего:',len(GRID),'траекторий ×',len(CHECKPOINTS),'checkpoints =',len(GRID)*len(CHECKPOINTS))
print('Новых обучений:', len(GRID)-sum(REUSE_F16.values()))
print('Новая папка результатов:',OUT)


RUN_GRID=True   # Для полного нового эксперимента должно оставаться True.

if RUN_GRID:
    total_start=time.perf_counter()
    for number,(f,mode) in enumerate(GRID,1):
        p=MODE_PARAMS[mode]
        stem=stem_for(f,mode)
        if f==16 and REUSE_F16[mode]:
            print(f'[{number:02d}/16] REUSE original f=16 {mode}: all 3 checkpoints',flush=True)
            continue
        metric_file=CACHE/f'{stem}_metrics.csv'
        log_file=CACHE/f'{stem}_training_log.csv'
        full=cached_parts(CACHE,stem)
        if all(path.is_file() for path in full):
            previous=pd.read_csv(metric_file)
            require(len(previous)==3 and set(previous.iteration)==set(CHECKPOINTS)
                    and not previous[METRICS].isna().any().any(),
                    f'Кеш {stem} имеет неполные метрики')
            print(f'[{number:02d}/16] SKIP completed: {stem}',flush=True)
            continue
        # Очистить исключительно незавершённые файлы НОВОГО эксперимента.
        for path in full:
            path.unlink(missing_ok=True)
        print(f'[{number:02d}/16] FIT: {stem}',flush=True)
        start=time.perf_counter()
        model=IALS(factors=int(f),confidence_mode=mode,alpha=p['alpha'],
                   regularization=p['regularization'],random_seed=SEED)
        rows=[]
        prior_it=0
        for it in CHECKPOINTS:
            if prior_it==0:
                model.fit(HISTORY,USER_IDS,ITEM_IDS,n_iters=it,
                          history_start_week=0,history_end_week=24)
            else:
                model.continue_fit(additional_iterations=it-prior_it)
            prior_it=it
            require(model.n_iterations==it and np.array_equal(model.user_ids,USER_IDS)
                    and np.array_equal(model.item_ids,ITEM_IDS),
                    f'{stem}: iALS изменил порядок ID или число итераций')
            scores=model.score()
            require(scores.shape==(3000,2000) and scores.dtype==np.float64
                    and np.isfinite(scores).all(), f'{stem}: неверные scores')
            ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
                               history=HISTORY,holdout=HOLDOUT,k=TOP_K,
                               history_start_week=0,history_end_week=24,holdout_week=25)
            s=ev.summary
            require(s['n_users']==3000 and s['n_eligible']==2082 and
                    s['n_history']==1454216 and s['n_holdout']==16158 and
                    s['holdout_week_checked'] is True and
                    len(ev.per_user)==3000 and len(ev.recommendations)==30000,
                    f'{stem}: повреждённые оценки')
            vals={
               'eligible_ndcg_at_10':float(s['eligible']['ndcg_at_10']),
               'eligible_recall_at_10':float(s['eligible']['recall_at_10']),
               'eligible_map_at_10':float(s['eligible']['map_at_10']),
               'all_users_ndcg_at_10':float(s['all_users']['ndcg_at_10']),
               'all_users_recall_at_10':float(s['all_users']['recall_at_10']),
               'all_users_map_at_10':float(s['all_users']['map_at_10']),
            }
            require(all(np.isfinite(v) and 0 <= v <= 1+1e-12 for v in vals.values()),
                    'Неверные метрики')
            rows.append({'factors':f,'alpha':p['alpha'],'regularization':p['regularization'],
                         'confidence_mode':mode,'iteration':it, **vals,
                         'eligible_share':float(s['eligible_share']),
                         'n_users':3000,'n_eligible':2082,
                         'train_loss':float(model.training_log[-1]['loss']),
                         'elapsed_seconds':float(time.perf_counter()-start)})
            ev.per_user.to_parquet(checkpoint_path(f,mode,it,'per_user'),index=False)
            ev.recommendations.to_parquet(checkpoint_path(f,mode,it,'recommendations'),index=False)
            print(f'   iteration {it:2d}: NDCG={vals["eligible_ndcg_at_10"]:.6f}; '
                  f'Recall={vals["eligible_recall_at_10"]:.6f}; '
                  f'MAP={vals["eligible_map_at_10"]:.6f}; '
                  f'loss={model.training_log[-1]["loss"]:.1f}',flush=True)
            del scores,ev
        log=pd.DataFrame(model.training_log)
        losses=log.loss.to_numpy()
        require(len(log)==15 and
                (np.diff(losses)<=1e-8*np.maximum(1.0,np.abs(losses[:-1]))).all(),
                f'{stem}: training loss вырос, проверьте численную устойчивость')
        log.insert(0,'confidence_mode',mode)
        log.insert(0,'regularization',p['regularization'])
        log.insert(0,'alpha',p['alpha'])
        log.insert(0,'factors',f)
        tmp_log=log_file.with_suffix('.tmp')
        log.to_csv(tmp_log,index=False)
        tmp_log.replace(log_file)
        # Маркер завершения metrics.csv записывается ПОСЛЕДНИМ, атомарно.
        tmp=metric_file.with_suffix('.tmp')
        pd.DataFrame(rows).to_csv(tmp,index=False)
        tmp.replace(metric_file)
        del model
        gc.collect()
    print(f'PASS: low-rank grid completed/reused in {time.perf_counter()-total_start:.1f} seconds')
else:
    print('RUN_GRID=False: новых результатов нет, выбор победителя запрещён')


require(RUN_GRID,'Нельзя анализировать незавершённый эксперимент')
rows=[]; logs=[]
for f,mode in GRID:
    stem=stem_for(f,mode)
    if f==16 and REUSE_F16[mode]:
        path=PREV_CACHE
    else:
        path=CACHE
    mp=path/f'{stem}_metrics.csv'
    lp=path/f'{stem}_training_log.csv'
    require(mp.is_file() and lp.is_file(),f'Незавершённая конфигурация {stem}')
    frame=pd.read_csv(mp)
    require(len(frame)==3 and set(frame.iteration)==set(CHECKPOINTS)
            and (frame.factors==f).all() and (frame.confidence_mode==mode).all(),
            f'Неверный набор checkpoints: {stem}')
    rows.append(frame)
    log=pd.read_csv(lp)
    require(len(log)==15,f'{stem}: неверное число loss checkpoints')
    logs.append(log)
LOW_RESULTS=pd.concat(rows,ignore_index=True)
LOW_LOGS=pd.concat(logs,ignore_index=True)
require(len(LOW_RESULTS)==48 and not LOW_RESULTS.duplicated(DIMS).any(),
        'Должны быть 48 уникальных validation-оценок')
require(len(LOW_LOGS)==16*15 and LOW_RESULTS[METRICS].notna().all().all(),
        'Неверная сводка экспериментов')
require((LOW_RESULTS.n_eligible==2082).all() and (LOW_RESULTS.n_users==3000).all(),
        'Размеры выборок различаются')

# Сравнение f=16 с первым экспериментом — не только лучшей точки, а всех 6.
for mode in CONFIDENCE_MODES:
    p=MODE_PARAMS[mode]
    old_part=OLD_RESULTS.loc[
        (OLD_RESULTS.factors==16)&(OLD_RESULTS.confidence_mode==mode)&
        (OLD_RESULTS.alpha==p['alpha'])&
        (OLD_RESULTS.regularization==p['regularization'])
    ].sort_values('iteration')
    new_part=LOW_RESULTS.loc[
        (LOW_RESULTS.factors==16)&(LOW_RESULTS.confidence_mode==mode)
    ].sort_values('iteration')
    require(len(old_part)==len(new_part)==3 and
            np.array_equal(old_part.iteration.to_numpy(),new_part.iteration.to_numpy()),
            'Не совпадают точки f=16 с исходным экспериментом')
    require(np.allclose(old_part[METRICS].to_numpy(),new_part[METRICS].to_numpy(),
                        atol=1e-10,rtol=1e-8),
            'f=16 дал другие метрики, нельзя объединять эксперименты')
print('PASS: обе f=16 траектории воспроизводят старые значения')

def sort_quality(frame):
    return frame.sort_values(
        ['eligible_ndcg_at_10','iteration','factors','alpha','regularization','confidence_mode'],
        ascending=[False,True,True,True,False,True],kind='mergesort')

# Для новых значений f добавляем прежние f=32/64 — только для честного общего выбора.
COMBINED=pd.concat([OLD_RESULTS,LOW_RESULTS],ignore_index=True)
COMBINED=COMBINED.drop_duplicates(DIMS,keep='last')
require(len(COMBINED)==72+42 and not COMBINED.duplicated(DIMS).any(),
        'Ожидалось 72 прежних + 42 НОВЫХ checkpoints (114 уникальных)')
BEST_LOW=sort_quality(LOW_RESULTS).iloc[0]
BEST_LOW_MODE=sort_quality(LOW_RESULTS).drop_duplicates('confidence_mode')
BEST_OLD=sort_quality(OLD_RESULTS).iloc[0]
BEST_ALL=sort_quality(COMBINED).iloc[0]
BEST_ALL_MODE=sort_quality(COMBINED).drop_duplicates('confidence_mode')

show_cols=DIMS+['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']
print('Лучшая low-rank конфигурация:')
display(BEST_LOW[show_cols].to_frame().T)
print('Лучшие конфигурации по confidence после расширения сетки:')
display(BEST_ALL_MODE[show_cols].sort_values('confidence_mode'))
print(f'Прошлый лучший iALS NDCG={BEST_OLD.eligible_ndcg_at_10:.6f}')
print(f'Лучший после low-rank NDCG={BEST_ALL.eligible_ndcg_at_10:.6f}')
print(f'Прирост = {BEST_ALL.eligible_ndcg_at_10-BEST_OLD.eligible_ndcg_at_10:+.6f} '
      f'({(BEST_ALL.eligible_ndcg_at_10/BEST_OLD.eligible_ndcg_at_10-1)*100:+.2f}%)')

BEST_AT_F=sort_quality(LOW_RESULTS).drop_duplicates(['confidence_mode','factors'])
print('Лучший checkpoint для каждого f и confidence:')
display(BEST_AT_F.sort_values(['confidence_mode','factors'])[show_cols])


import plotly.graph_objects as go

figures={}
# Best checkpoints at fixed params; markers identify whether best at 5/10/15.
plot_best=BEST_AT_F.sort_values('factors').copy()
fig=px.line(plot_best,x='factors',y='eligible_ndcg_at_10',color='confidence_mode',
            markers=True,custom_data=['iteration','alpha','regularization'],
            title='Low-rank iALS: лучший Eligible NDCG@10 при каждом f',
            labels={'factors':'Размерность f','eligible_ndcg_at_10':'Eligible NDCG@10',
                    'confidence_mode':'Confidence'})
fig.add_hline(y=float(BEST_OLD.eligible_ndcg_at_10),line_dash='dash',
              annotation_text='Старый лучший iALS (f=16)',annotation_position='top left')
fig.add_hline(y=0.018522044,line_dash='dot',line_color='gray',
              annotation_text='ItemKNN Binary k=640 (предыдущая validation)',
              annotation_position='bottom left')
fig.update_traces(hovertemplate='f=%{x}<br>NDCG=%{y:.6f}<br>iter=%{customdata[0]}'
                  '<br>alpha=%{customdata[1]}<br>lambda=%{customdata[2]}<extra>%{fullData.name}</extra>')
fig.update_xaxes(tickmode='array',tickvals=list(FACTORS))
fig.show(); figures['ndcg_best_by_f']=fig

for mode in CONFIDENCE_MODES:
    subset=LOW_RESULTS.loc[LOW_RESULTS.confidence_mode==mode].copy()
    subset['checkpoint']=subset.iteration.map(lambda v:f'{int(v)} итераций')
    fig=px.line(subset.sort_values('factors'),x='factors',y='eligible_ndcg_at_10',
                color='checkpoint',markers=True,
                title=f'{mode.capitalize()}: NDCG@10 по f и числу ALS-итераций',
                labels={'factors':'Размерность f','eligible_ndcg_at_10':'Eligible NDCG@10',
                        'checkpoint':'Checkpoint'})
    fig.update_xaxes(tickmode='array',tickvals=list(FACTORS))
    fig.show(); figures[f'ndcg_by_checkpoint_{mode}']=fig

for metric,title in [('eligible_recall_at_10','Recall@10'),
                     ('eligible_map_at_10','MAP@10')]:
    fig=px.line(plot_best,x='factors',y=metric,color='confidence_mode',markers=True,
                title=f'{title} для checkpoints, выбранных по NDCG@10',
                labels={'factors':'Размерность f',metric:title,'confidence_mode':'Confidence'},
                custom_data=['iteration'])
    fig.update_xaxes(tickmode='array',tickvals=list(FACTORS))
    fig.show(); figures[metric]=fig

# В старом эксперименте с f=32/64 alpha и lambda подбирались свободно.
old_by_f=sort_quality(OLD_RESULTS).drop_duplicates(['factors','confidence_mode'])
old_by_f=old_by_f.assign(series=lambda x:x.confidence_mode+' · старая полная сетка')
new_by_f=plot_best.assign(series=lambda x:x.confidence_mode+' · фиксированные α/λ')
context=pd.concat([old_by_f,new_by_f],ignore_index=True)
fig=px.line(context.sort_values('factors'),x='factors',y='eligible_ndcg_at_10',
            color='series',markers=True,custom_data=['iteration','alpha','regularization'],
            title='Контекст: low-rank и прежние f=16/32/64 (разный объём настройки)',
            labels={'factors':'Размерность f','eligible_ndcg_at_10':'Eligible NDCG@10','series':'Серия'})
fig.update_traces(hovertemplate='f=%{x}; NDCG=%{y:.6f}; iter=%{customdata[0]}; '
                  'alpha=%{customdata[1]}; lambda=%{customdata[2]}<extra>%{fullData.name}</extra>')
fig.show(); figures['context_old_vs_lowrank']=fig


# Прежний победитель был сохранён в review_bundle отдельной таблицей.
old_mode=str(BEST_OLD.confidence_mode)
old_winner_meta=pd.read_csv(PREV_REVIEW/'best_binary_and_graded.csv')
old_mode_winner=old_winner_meta.loc[old_winner_meta.confidence_mode==old_mode]
require(len(old_mode_winner)==1 and
        all(old_mode_winner.iloc[0][key]==BEST_OLD[key] for key in DIMS),
        'Прежний выбранный победитель не совпадает с сохранённым per_user')
old_pu=pd.read_csv(PREV_REVIEW/f'winner_{old_mode}_per_user.csv',dtype={'user_id':'uint64'})

# Для лучшего нового/общего варианта берём идентичные user axes из сохранённого кеша.
def read_candidate_per_user(row):
    f=int(row.factors); mode=str(row.confidence_mode); it=int(row.iteration)
    if f in FACTORS and abs(row.alpha-MODE_PARAMS[mode]['alpha'])<1e-12 and abs(row.regularization-MODE_PARAMS[mode]['regularization'])<1e-12:
        file=checkpoint_path(f,mode,it,'per_user')
        if file.is_file():
            return pd.read_parquet(file)
    ref=old_winner_meta.loc[old_winner_meta.confidence_mode==mode]
    if len(ref)==1 and all(ref.iloc[0][key]==row[key] for key in DIMS):
        return pd.read_csv(PREV_REVIEW/f'winner_{mode}_per_user.csv',dtype={'user_id':'uint64'})
    raise FileNotFoundError('Нет подтверждённого per-user для выбранной конфигурации')

best_pu=read_candidate_per_user(BEST_ALL)
require(len(old_pu)==len(best_pu)==3000 and
        int(old_pu.eligible.sum())==int(best_pu.eligible.sum())==2082,
        'Персональные метрики должны быть на одних и тех же пользователях')
paired=[]; intervals=[]
for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
    paired.append({'metric':metric,
                   **bootstrap_paired_difference(best_pu,old_pu,metric=metric,
                                                group='eligible',n_bootstrap=BOOTSTRAP_N,
                                                ci=.95,seed=BOOTSTRAP_SEED)})
    intervals.append({'metric':metric,'model':'best_after_lowrank',
                      **bootstrap_mean(best_pu,metric=metric,group='eligible',
                                      n_bootstrap=BOOTSTRAP_N,ci=.95,seed=BOOTSTRAP_SEED)})
    intervals.append({'metric':metric,'model':'previous_best',
                      **bootstrap_mean(old_pu,metric=metric,group='eligible',
                                      n_bootstrap=BOOTSTRAP_N,ci=.95,seed=BOOTSTRAP_SEED)})
PAIRED=pd.DataFrame(paired)
INTERVALS=pd.DataFrame(intervals)
print('Лучший после расширения − прежний лучший:')
display(PAIRED[['metric','estimate','ci_low','ci_high']])
print('CI каждого выбранного варианта (описательные, post-selection):')
display(INTERVALS[['model','metric','estimate','ci_low','ci_high']])

# Дополнительно проверим, что разница в NDCG совпадает с сводкой.
ndcg_delta=float(BEST_ALL.eligible_ndcg_at_10-BEST_OLD.eligible_ndcg_at_10)
require(np.isclose(PAIRED.loc[PAIRED.metric=='ndcg_at_10','estimate'].iloc[0],
                   ndcg_delta,atol=1e-12), 'Paired NDCG не совпадает с общей разницей')


REVIEW.mkdir(parents=True,exist_ok=True)
LOW_RESULTS.sort_values(DIMS).to_csv(REVIEW/'lowrank_48_metrics.csv',index=False)
LOW_LOGS.sort_values(['confidence_mode','factors','iteration']).to_csv(
    REVIEW/'lowrank_training_loss.csv',index=False)
COMBINED.sort_values(DIMS).to_csv(REVIEW/'all_114_combined_metrics.csv',index=False)
BEST_AT_F.sort_values(['confidence_mode','factors']).to_csv(REVIEW/'best_checkpoint_per_f.csv',index=False)
BEST_ALL_MODE.to_csv(REVIEW/'best_per_mode_after_extension.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_new_best_minus_previous_best.csv',index=False)
INTERVALS.to_csv(REVIEW/'bootstrap_best_and_previous.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)

for mode in CONFIDENCE_MODES:
    row=BEST_ALL_MODE.loc[BEST_ALL_MODE.confidence_mode==mode].iloc[0]
    pu=read_candidate_per_user(row)
    pu.to_csv(REVIEW/f'winner_{mode}_per_user.csv',index=False)
    f=int(row.factors); it=int(row.iteration)
    rec=checkpoint_path(f,mode,it,'recommendations')
    if rec.is_file() and abs(row.alpha-MODE_PARAMS[mode]['alpha'])<1e-12 and abs(row.regularization-MODE_PARAMS[mode]['regularization'])<1e-12:
        recommendations=pd.read_parquet(rec)
    else:
        # Эта ветка соответствует прежнему лучшему в случае, если он не из lowrank-среза.
        previous=old_winner_meta.loc[old_winner_meta.confidence_mode==mode].iloc[0]
        require(all(previous[key]==row[key] for key in DIMS),
                'Не удаётся восстановить Top-10 для победителя')
        recommendations=pd.read_csv(PREV_REVIEW/f'winner_{mode}_recommendations.csv',
                                    dtype={'user_id':'uint64','item_id':'uint64'})
    require(len(pu)==3000 and len(recommendations)==30000,
            'Неверное число per-user или рекомендаций')
    require(not pd.MultiIndex.from_frame(recommendations[['user_id','item_id']]).isin(
            pd.MultiIndex.from_frame(HISTORY[['user_id','item_id']])).any(),
            'Найдены просмотренные видео в Top-10 победителя')
    recommendations.to_csv(REVIEW/f'winner_{mode}_recommendations.csv',index=False)

# Plotly: интерактивные HTML с внешней библиотекой (компактный архив).
for label,fig in figures.items():
    fig.write_html(str(REVIEW/f'{label}.html'),include_plotlyjs='cdn',full_html=True)

selected={
    'old_best':{key:(BEST_OLD[key].item() if hasattr(BEST_OLD[key],'item') else BEST_OLD[key]) for key in DIMS+METRICS[:3]},
    'lowrank_best':{key:(BEST_LOW[key].item() if hasattr(BEST_LOW[key],'item') else BEST_LOW[key]) for key in DIMS+METRICS[:3]},
    'best_after_extension':{key:(BEST_ALL[key].item() if hasattr(BEST_ALL[key],'item') else BEST_ALL[key]) for key in DIMS+METRICS[:3]},
}
manifest={
    'status':'PASS_VALIDATION_ONLY','experiment':'iALS low-rank study',
    'source_revision':REVISION['revision'], 'training_weeks':'0-24','validation_week':25,
    'train_rows':len(HISTORY),'validation_rows':len(HOLDOUT),
    'n_users':3000,'n_items':2000,'n_eligible':2082,
    'input_file_sha256':file_hashes,'module_sha256':module_hashes,
    'user_ids_sha256':user_hash,'item_ids_sha256':item_hash,
    'first_experiment_sha256':{'run_identity':sha256_file(old_id_path),
                               'all_72_validation_metrics':sha256_file(old_csv_path)},
    'lowrank_factors':list(FACTORS),'mode_params':MODE_PARAMS,
    'iterations_evaluated':list(CHECKPOINTS),'validation_rows_total':48,
    'new_validation_checkpoints':42 if all(REUSE_F16.values()) else 42+3*sum(not x for x in REUSE_F16.values()),
    'original_validation_rows':72,'unique_combined_configurations':114,
    'reused_original_f16':REUSE_F16,
    'selection_metric':'eligible_ndcg_at_10, linear gains rel 0/1/2',
    'selection':selected,
    'bootstrap':{'resamples':BOOTSTRAP_N,'seed':BOOTSTRAP_SEED,'paired_by_user':True,
                 'post_selection_bias_warning':True},
    'test_data_read':False,'test_metrics_calculated':False,
    'read_previous_test_results':False,
}
(REVIEW/'selection_and_manifest.json').write_text(
    json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
(REVIEW/'README_results.md').write_text(
    '# VK-LSVD — Low-rank iALS validation extension\n\n'
    'Train 0–24, validation 25, test untouched. Fixed binary alpha=40,lambda=10; '
    'graded alpha=10,lambda=1. Factors 2,4,6,8,10,12,14,16; checkpoints 5,10,15. '
    'Previous 72 configurations preserved. Best selected on Eligible NDCG@10 only. '
    'Post-selection bootstrap is descriptive. user_id/item_id require uint64-safe reading. '
    'HTML charts use Plotly CDN for a smaller archive; Python notebook charts are interactive.\n',
    encoding='utf-8')
require(len(list(REVIEW.glob('winner_*_per_user.csv')))==2 and
        len(list(REVIEW.glob('winner_*_recommendations.csv')))==2,
        'Не все финальные файлы сохранены')
print('PASS: review bundle готов:',REVIEW)
print('Результаты сохранены.')
print('TEST НЕ ЧИТАЛСЯ.')
