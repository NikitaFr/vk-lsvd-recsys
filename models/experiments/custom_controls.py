"""custom controls experiment on train/validation."""
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

for name in ("pyarrow", "scipy", "plotly", "nbformat"):
    if importlib.util.find_spec(name) is None:
        raise ModuleNotFoundError(
            f"В текущем Jupyter kernel отсутствует {name}. "
            f"Установи через %pip install {name}, перезапусти kernel и запусти Run All."
        )
import scipy
import plotly
import plotly.express as px
import plotly.graph_objects as go
from itertools import product

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
OUT = ROOT / "results" / "ials" / "custom_target_diagnostics"

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

# Новая диагностика: все гиперпараметры ALS фиксированы, меняем лишь target/seed/итерации.
FACTORS = 12
ALPHA = 40.0
REGULARIZATION = 10.0
SEEDS = (42, 43, 44)
CHECKPOINTS = (10, 15)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
TOP_K = 10
WINNER_PARAMS = {"q_min": 0.4, "watch_weight": 0.3, "delta": 0.0, "tau": 60, "gamma": 0.1}
EXPERIMENTS = {
    "winner": WINNER_PARAMS,
    "time_only": {**WINNER_PARAMS, "watch_weight": 0.0},
    "no_explicit_bonus": {**WINNER_PARAMS, "gamma": 0.0},
    "tau_120": {**WINNER_PARAMS, "tau": 120},
}
RUNS = [(name,seed) for name in EXPERIMENTS for seed in (SEEDS if name=="winner" else (42,))]
RUNS += [("binary_control", seed) for seed in SEEDS]
assert len(RUNS)==9 and len(set(RUNS))==9


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
require(len(EXPERIMENTS)==4 and len(RUNS)==9 and CHECKPOINTS==(10,15),
        "Некорректные диагностические конфигурации")
require(WINNER_PARAMS == {"q_min":.4,"watch_weight":.3,"delta":.0,"tau":60,"gamma":.1},
        "Изменены параметры исходного победителя")
require(FACTORS==12 and ALPHA==40. and REGULARIZATION==10. and SEEDS==(42,43,44), "Базовые параметры iALS изменены")
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
# IMPORTANT: TRAIN_SIGNALS и HISTORY имеют ИДЕНТИЧНЫЙ порядок строк.
# В target не допускается обращение к val_labeled / HOLDOUT.
TRAIN_SIGNALS = train_labeled[['timespent','duration','like','bookmark','dislike','rel']].copy()
require(len(TRAIN_SIGNALS)==len(HISTORY), 'Target features and history misaligned')
del train_raw, validation_raw, train_joined, val_joined, train_labeled, val_labeled, items_meta
# release extra references to large frames before fitting
_ = gc.collect()


def continuous_target(events, *, q_min, watch_weight, delta, tau, gamma=0.1):
    """Only TRAIN observed events. Row order unchanged; no validation data, no ID-based joins."""
    assert 0 <= q_min < 1 and 0 <= watch_weight <= 1 and 0 <= delta <= 1
    assert tau > 0 and gamma >= 0
    t = events['timespent'].to_numpy(dtype=np.float64)
    d = events['duration'].to_numpy(dtype=np.float64)
    rel = events['rel'].to_numpy(dtype=np.int8)
    e = (events['like'].to_numpy() == 1) | (events['bookmark'].to_numpy() == 1)
    dislike = events['dislike'].to_numpy() == 1
    require(np.isfinite(t).all() and np.isfinite(d).all() and (t >= 0).all() and (d > 0).all(), 'Bad times/durations')
    require(np.isin(rel, [0,1,2]).all(), 'Bad relevance grades')
    with np.errstate(over='ignore'):
        r = t/d
    fr = ((1-delta)*np.minimum(r, 1.0) + delta*np.minimum(np.maximum(r-1.0, 0.0), 1.0))
    ft = np.log1p(np.minimum(t, float(tau))) / np.log1p(float(tau))
    raw = q_min + (1-q_min) * (watch_weight*fr + (1-watch_weight)*ft) + gamma*e.astype(float)
    out = np.where(rel > 0, np.minimum(raw, 1.0), 0.0).astype(np.float64)
    require(np.isfinite(out).all() and ((out >= 0) & (out <= 1)).all(), 'Target must be finite in [0,1]')
    require(np.all(out[rel == 0] == 0.0), 'rel=0 must map to zero')
    require(np.all(out[rel > 0] >= q_min-1e-12), 'Positive below q_min')
    require(np.all(out[dislike] == 0.), 'Dislike must override other signals')
    return out

# Counterexamples to common mistakes: dislike priority, >100% watch and explicit feedback.
toy = pd.DataFrame({'timespent':[9.,20.,3.,80.,50.], 'duration':[10.,10.,120.,100.,100.],
    'like':[0,0,1,0,1], 'bookmark':[0,0,0,0,0], 'dislike':[0,0,0,1,0], 'rel':[2,2,2,0,2]})
qt = continuous_target(toy, q_min=.5, watch_weight=.6, delta=.2, tau=60, gamma=.1)
require(len(qt)==5 and qt[3]==0.0 and qt[2]>=.5 and qt[1]>qt[0], 'Toy oracle failed')
# Independent exact check of one edge case.
first = .5+.5*(.6*(.8*.9)+.4*(np.log1p(9)/np.log1p(60)))
require(np.isclose(qt[0],first,atol=1e-13), 'Independent formula check failed')
print('Target toy checks: PASS', np.round(qt,5))


tiny_u=np.array([2**63+5,2**63+6],dtype=np.uint64)
tiny_i=np.arange(10,26,dtype=np.uint64)
tiny_h=pd.DataFrame({'user_id':np.array([tiny_u[0],tiny_u[0],tiny_u[1]],dtype=np.uint64),
                     'item_id':np.array([tiny_i[0],tiny_i[1],tiny_i[1]],dtype=np.uint64),
                     'week':[0,0,0],'rel':[0,2,1]})
tiny_hold=pd.DataFrame({'user_id':np.array([tiny_u[0],tiny_u[1]],dtype=np.uint64),
                       'item_id':np.array([tiny_i[2],tiny_i[3]],dtype=np.uint64),
                       'week':[1,1],'rel':[2,1]})
tiny_target=np.array([0.,.75,.55])
tiny_model=IALS(factors=2,regularization=1.,alpha=5.,confidence_mode='binary',random_seed=42)
tiny_model.fit(tiny_h,tiny_u,tiny_i,n_iters=2,observed_targets=tiny_target,
               history_start_week=0,history_end_week=0)
require(tiny_model.score().shape==(2,16) and np.isfinite(tiny_model.score()).all(), 'Tiny ALS failed')
tiny_ev=evaluate_scores(tiny_model.score(),tiny_u,tiny_i,tiny_h,tiny_hold,k=10,
                        history_start_week=0,history_end_week=0,holdout_week=1)
require(len(tiny_ev.per_user)==2 and len(tiny_ev.recommendations)==20, 'Tiny evaluator failed')
print('PASS: custom targets + existing iALS normal equations + common evaluator')


OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / 'checkpoint_cache'
REVIEW = OUT / 'review_bundle'
CACHE.mkdir(parents=True, exist_ok=True)

RUN_IDENTITY = {
    'experiment': 'VK_LSVD_ials_custom_target_diagnostics_v1.0',
    'formula': 'same_continuous_target_as_24_grid_v1.0',
    'file_sha256': file_hashes,
    'module_sha256': module_hashes,
    'id_sha256': {'users':user_hash,'items':item_hash},
    'winner_params': WINNER_PARAMS,
    'experiments': EXPERIMENTS,
    'runs': [{'variant':v,'seed':int(seed)} for v,seed in RUNS],
    'ials': {'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
             'confidence_mode':'binary','iterations':list(CHECKPOINTS)},
    'protocol': {'train':'0..24','validation':25,'k':TOP_K,'ndcg_gains':'linear rel 0/1/2',
                 'test_read':False}
}
ident = OUT / 'run_identity.json'
if ident.exists():
    require(json.loads(ident.read_text(encoding='utf-8')) == RUN_IDENTITY,
            'Идентичность кэша не совпала! Используй новую папку; не смешивай результаты.')
else:
    require(not any(CACHE.iterdir()), 'Найдены cache-файлы без run_identity.json')
    ident.write_text(json.dumps(RUN_IDENTITY, indent=2, ensure_ascii=False), encoding='utf-8')

# В имени ключа явно фиксируется seed, чтобы seed=43 не подменил seed=42.
def stem_for(variant, seed):
    require(variant in EXPERIMENTS or variant == 'binary_control', 'Unknown variant')
    require(int(seed) in SEEDS, 'Unknown seed')
    return f'{variant}_seed{int(seed)}'

def per_path(stem, iteration):
    return CACHE / f'{stem}_it{iteration}_per_user.parquet'

def rec_path(stem, iteration):
    return CACHE / f'{stem}_it{iteration}_recommendations.parquet'

def atomic_csv(frame, path):
    tmp = path.with_name(path.name + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)

def atomic_parquet(frame, path):
    tmp = path.with_name(path.name + '.tmp.parquet')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)

def evaluate_model(model):
    scores=model.score()
    ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,history=HISTORY,
        holdout=HOLDOUT,k=TOP_K,history_start_week=0,history_end_week=24,holdout_week=25)
    ss=ev.summary
    require(ss['n_users']==3000 and ss['n_eligible']==2082 and
            ss['n_history']==1454216 and ss['n_holdout']==16158, 'Evaluator protocol mismatch')
    return ev

def metrics_from_eval(ev):
    s=ev.summary
    return {'eligible_ndcg_at_10':float(s['eligible']['ndcg_at_10']),
            'eligible_recall_at_10':float(s['eligible']['recall_at_10']),
            'eligible_map_at_10':float(s['eligible']['map_at_10']),
            'all_users_ndcg_at_10':float(s['all_users']['ndcg_at_10']),
            'all_users_recall_at_10':float(s['all_users']['recall_at_10']),
            'all_users_map_at_10':float(s['all_users']['map_at_10'])}
print('Diagnostic runs',len(RUNS),'× checkpoints',CHECKPOINTS,'=',len(RUNS)*len(CHECKPOINTS),'evaluations')


require(len(RUNS) == 9 and len(CHECKPOINTS)==2, 'Wrong run count')
TARGETS = {name: continuous_target(TRAIN_SIGNALS, **params) for name,params in EXPERIMENTS.items()}
rel_vec = TRAIN_SIGNALS.rel.to_numpy()
for name, q in TARGETS.items():
    require(q.shape == (len(HISTORY),) and np.isfinite(q).all(), f'{name}: bad target shape')
    require(np.array_equal(q[rel_vec == 0], np.zeros(np.count_nonzero(rel_vec == 0))),
            f'{name}: rel=0 target mismatch')
    print(name, 'target_mean_positive=', round(float(q[rel_vec>0].mean()),6),
          'share_saturated=', round(float(np.mean(q[rel_vec>0]>=1)),4))

for run_idx, (variant, seed) in enumerate(RUNS, 1):
    stem = stem_for(variant, seed)
    done = CACHE / (stem + '_metrics.csv')
    logpath = CACHE / (stem + '_train_loss.csv')
    artifacts = [p for iteration in CHECKPOINTS
                 for p in (per_path(stem,iteration),rec_path(stem,iteration))]
    if done.exists() and logpath.exists() and all(p.exists() for p in artifacts):
        cached = pd.read_csv(done)
        require(len(cached)==len(CHECKPOINTS) and
                set(cached.iteration)==set(CHECKPOINTS) and
                set(cached.seed)=={seed} and set(cached.variant)=={variant},
                f'Incorrect cache entry: {stem}')
        print(f'[{run_idx}/{len(RUNS)}] CACHED {stem}', flush=True)
        continue
    # У iALS нет штатного восстановления факторов после прерывания. Частичная
    # траектория честно начинается заново с тем же seed; другие готовые прогоны не трогаем.
    done.unlink(missing_ok=True); logpath.unlink(missing_ok=True)
    for path in artifacts: path.unlink(missing_ok=True)
    q = None if variant == 'binary_control' else TARGETS[variant]
    q_pos = ((rel_vec>0).astype(float) if q is None else q[rel_vec>0])
    if q is None:
        q_avg = 1.0; q_sat = 1.0
    else:
        q_avg = float(q_pos.mean()); q_sat = float(np.mean(q_pos>=1))
    model = IALS(factors=FACTORS, alpha=ALPHA, regularization=REGULARIZATION,
                 confidence_mode='binary', random_seed=int(seed))
    completed = 0; rows = []; start=time.perf_counter()
    print(f'[{run_idx}/{len(RUNS)}] TRAIN {stem}', flush=True)
    for it in CHECKPOINTS:
        if completed == 0:
            model.fit(HISTORY,USER_IDS,ITEM_IDS,n_iters=it,observed_targets=q,
                      history_start_week=0,history_end_week=24)
        else:
            model.continue_fit(additional_iterations=it-completed)
        completed=it
        ev=evaluate_model(model)
        row={'variant':variant,'seed':int(seed),'iteration':int(it),
             'q_min':np.nan if q is None else EXPERIMENTS[variant]['q_min'],
             'w':np.nan if q is None else EXPERIMENTS[variant]['watch_weight'],
             'delta':np.nan if q is None else EXPERIMENTS[variant]['delta'],
             'tau':np.nan if q is None else EXPERIMENTS[variant]['tau'],
             'gamma':np.nan if q is None else EXPERIMENTS[variant]['gamma'],
             'target_mean_positive':q_avg,'target_saturated_positive':q_sat,
             'elapsed_seconds':float(time.perf_counter()-start),
             'train_loss':float(model.training_log[-1]['loss']), **metrics_from_eval(ev)}
        rows.append(row)
        atomic_parquet(ev.per_user, per_path(stem,it))
        atomic_parquet(ev.recommendations, rec_path(stem,it))
        print(f'  iter={it}: NDCG {row["eligible_ndcg_at_10"]:.6f}, '
              f'Recall {row["eligible_recall_at_10"]:.6f}, MAP {row["eligible_map_at_10"]:.6f}',flush=True)
        del ev
    log=pd.DataFrame(model.training_log)
    require(len(log)==max(CHECKPOINTS) and np.isfinite(log.loss).all(),f'Invalid ALS loss: {stem}')
    log.insert(0,'run_id',stem); log.insert(1,'variant',variant); log.insert(2,'seed',int(seed))
    atomic_csv(log,logpath)
    atomic_csv(pd.DataFrame(rows),done)  # completion marker LAST
    del model, q_pos
    gc.collect()
print('PASS: all 9 runs completed or securely cached')


frames=[]; logs=[]
for variant,seed in RUNS:
    stem=stem_for(variant,seed)
    metrics_file=CACHE/(stem+'_metrics.csv')
    require(metrics_file.exists(), f'Missing run {stem}')
    part=pd.read_csv(metrics_file)
    require(len(part)==2 and set(part.iteration)==set(CHECKPOINTS),f'Wrong results {stem}')
    frames.append(part)
    logs.append(pd.read_csv(CACHE/(stem+'_train_loss.csv')))
RESULTS = pd.concat(frames, ignore_index=True)
TRAIN_LOGS = pd.concat(logs, ignore_index=True)
require(len(RESULTS)==18 and not RESULTS.duplicated(['variant','seed','iteration']).any(),
        'Expected 18 unique model evaluations')
RESULTS = RESULTS.sort_values(['variant','seed','iteration']).reset_index(drop=True)
print('ALL validation results:')
display(RESULTS[['variant','seed','iteration','eligible_ndcg_at_10',
                 'eligible_recall_at_10','eligible_map_at_10','train_loss']].round(6))
# We do not choose the best seed. Average over the PREDECLARED three seeds, same weights.
SEED_STATS=(RESULTS.loc[RESULTS.variant.isin(['winner','binary_control'])]
            .groupby(['variant','iteration'],as_index=False)
            .agg(mean_ndcg=('eligible_ndcg_at_10','mean'),
                 std_ndcg=('eligible_ndcg_at_10','std'),
                 min_ndcg=('eligible_ndcg_at_10','min'),
                 max_ndcg=('eligible_ndcg_at_10','max'),
                 mean_recall=('eligible_recall_at_10','mean'),
                 mean_map=('eligible_map_at_10','mean'),
                 n_seeds=('seed','nunique')))
require((SEED_STATS.n_seeds==3).all(), 'Not exactly three seeds per condition')
print('SEED ROBUSTNESS (unweighted mean over seeds):')
display(SEED_STATS.round(6))
ABLATION=(RESULTS.loc[(RESULTS.seed==42)&(RESULTS.variant.isin(EXPERIMENTS))]
          .copy())
reference=(ABLATION.loc[ABLATION.variant=='winner',['iteration','eligible_ndcg_at_10']]
           .set_index('iteration').eligible_ndcg_at_10)
ABLATION['delta_vs_winner_ndcg']=[v-reference.loc[it] for v,it in
                                  zip(ABLATION.eligible_ndcg_at_10,ABLATION.iteration)]
print('ABLATIONS vs winner, matching seed=42 and checkpoint:')
display(ABLATION[['variant','iteration','eligible_ndcg_at_10','delta_vs_winner_ndcg']].round(6))
# Reproduction guard: previously accepted 10-iteration winner and binary baseline.
win42=RESULTS.query('variant=="winner" and seed==42 and iteration==10').iloc[0]
base42=RESULTS.query('variant=="binary_control" and seed==42 and iteration==10').iloc[0]
require(abs(win42.eligible_ndcg_at_10-0.026986) < 0.000020,
        'Winner 10 iterations seed42 differs from the previous validation; stop')
require(abs(base42.eligible_ndcg_at_10-0.020702) < 0.000020,
        'Binary 10 iterations seed42 differs from original iALS; stop')
print('PASS: baseline and winner reproduced at seed42/iteration10')


PAIRED_ROWS=[]
def paired_comparison(variant_a,seed_a,it_a,variant_b,seed_b,it_b,reason):
    first=pd.read_parquet(per_path(stem_for(variant_a,seed_a),it_a))
    second=pd.read_parquet(per_path(stem_for(variant_b,seed_b),it_b))
    require(len(first)==len(second)==len(USER_IDS), 'Per-user tables misaligned')
    for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
        result=bootstrap_paired_difference(first,second,metric=metric,group='eligible',
                                            n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        PAIRED_ROWS.append({'comparison':reason,'variant_a':variant_a,'seed_a':seed_a,
                            'iteration_a':it_a,'variant_b':variant_b,'seed_b':seed_b,
                            'iteration_b':it_b,'metric':metric,**result})

for it in CHECKPOINTS:
    # Ablations vs same-seed winner.
    for variant in ('time_only','no_explicit_bonus','tau_120'):
        paired_comparison(variant,42,it,'winner',42,it,f'{variant}_minus_winner_it{it}')
    # Winner 10 vs 15 within seed (only compare SAME SEED).
    if it==15:
        for seed in SEEDS:
            paired_comparison('winner',seed,15,'winner',seed,10,
                              f'winner_15_minus_10_seed{seed}')
    # Winner vs binary, separately for each seed, no winner selection by seed.
    for seed in SEEDS:
        paired_comparison('winner',seed,it,'binary_control',seed,it,
                          f'winner_minus_binary_seed{seed}_it{it}')
PAIRED=pd.DataFrame(PAIRED_ROWS)
require(len(PAIRED)==45,'Expected 15 comparisons × 3 metrics')
display(PAIRED.loc[PAIRED.metric=='ndcg_at_10',
                   ['comparison','estimate','ci_low','ci_high']].round(6))
print('Reminder: user-level bootstrap on tuned validation does NOT remove selection bias.')


FIGURES={}
fig=px.line(RESULTS.loc[(RESULTS.variant.isin(EXPERIMENTS))&(RESULTS.seed==42)],
            x='iteration',y='eligible_ndcg_at_10',color='variant',markers=True,
            title='Custom iALS: проверка компонентов target (seed=42)',
            labels={'iteration':'Итерации ALS','eligible_ndcg_at_10':'Eligible NDCG@10'})
fig.update_layout(height=490);fig.show();FIGURES['ablation_ndcg']=fig
fig=px.line(RESULTS.loc[RESULTS.variant=='winner'], x='iteration', y='eligible_ndcg_at_10',
            color=RESULTS.loc[RESULTS.variant=='winner','seed'].astype(str),markers=True,
            title='Устойчивость победителя по seeds',
            labels={'iteration':'Итерации ALS','eligible_ndcg_at_10':'Eligible NDCG@10',
                    'color':'Seed'})
fig.update_layout(height=490);fig.show();FIGURES['winner_seed_ndcg']=fig
fig=px.line(SEED_STATS,x='iteration',y='mean_ndcg',color='variant',markers=True,
            error_y='std_ndcg',title='Средние NDCG@10 ± std между seeds (n=3)',
            labels={'iteration':'Итерации ALS','mean_ndcg':'Средняя Eligible NDCG@10'})
fig.update_layout(height=490);fig.show();FIGURES['seed_mean_std_ndcg']=fig
fig=px.line(TRAIN_LOGS.loc[TRAIN_LOGS.seed==42],x='iteration',y='loss',color='variant',
            title='Тренировочная функция потерь (seed=42)',
            labels={'iteration':'Итерации ALS','loss':'Weighted ALS loss'})
fig.update_layout(height=490);fig.show();FIGURES['train_loss']=fig


REVIEW.mkdir(parents=True,exist_ok=True)
RESULTS.to_csv(REVIEW/'all_18_validation_metrics.csv',index=False)
TRAIN_LOGS.to_csv(REVIEW/'all_training_iterations.csv',index=False)
SEED_STATS.to_csv(REVIEW/'seed_mean_std.csv',index=False)
ABLATION.to_csv(REVIEW/'ablation_vs_winner.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_differences.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
# All winner seeds, at each checkpoint: necessary for review and matched-user comparisons.
for seed in SEEDS:
    for it in CHECKPOINTS:
        stem=stem_for('winner',seed)
        pd.read_parquet(per_path(stem,it)).to_csv(REVIEW/f'winner_seed{seed}_it{it}_per_user.csv',index=False)
# Only winner seed42 recs, for concentration audit. Data volume controlled.
for it in CHECKPOINTS:
    stem=stem_for('winner',42)
    pd.read_parquet(rec_path(stem,it)).to_csv(REVIEW/f'winner_seed42_it{it}_recommendations.csv',index=False)
# Separate per-user tables for all ablation and control runs for independent verification of paired bootstrap.
for variant,seed in RUNS:
    if variant=='winner': continue
    stem=stem_for(variant,seed)
    for it in CHECKPOINTS:
        pd.read_parquet(per_path(stem,it)).to_csv(REVIEW/f'{stem}_it{it}_per_user.csv',index=False)
for name,fig in FIGURES.items():
    fig.write_html(str(REVIEW/(name+'.html')), include_plotlyjs=True, full_html=True)
# Store only TRAIN-target distributions.
target_summary=[]
for variant,q in TARGETS.items():
    for rel in (0,1,2):
        sub=q[rel_vec==rel]
        target_summary.append({'variant':variant,'rel':rel,'count':int(len(sub)),
                               'mean':float(sub.mean()),'p10':float(np.quantile(sub,.1)),
                               'median':float(np.median(sub)),'p90':float(np.quantile(sub,.9)),
                               'max':float(sub.max())})
pd.DataFrame(target_summary).to_csv(REVIEW/'target_distribution_by_rel.csv',index=False)
manifest={
    'experiment':'VK-LSVD Custom iALS final target diagnostics and seed robustness v1.0',
    'file_sha256':file_hashes,'module_sha256':module_hashes,
    'ids_sha256':{'users':user_hash,'items':item_hash},
    'dataset':{'train_weeks':'0..24','validation_week':25,'test_read':False,
               'train_rows':int(len(HISTORY)),'validation_rows':int(len(HOLDOUT)),
               'users':len(USER_IDS),'items':len(ITEM_IDS)},
    'ials':{'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
            'confidence_mode':'binary','seeds':list(SEEDS),'iterations':list(CHECKPOINTS)},
    'target_params':EXPERIMENTS,
    'comparison':'per-user paired bootstrap, n_bootstrap=2000, 95% percentile CI',
    'validation_selection_bias_warning':True,
    'unknown_pairs':'all unobserved user-item pairs preference=0 confidence=1',
    'metric':'Eligible NDCG@10 with linear rel gains',
    'previous_frozen_winner':{'seed':42,'iterations':10,'ndcg_at_10_approx':0.026986},
    'not_choose_best_seed_on_test':True,
}
(REVIEW/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False), encoding='utf-8')
(REVIEW/'README_results.md').write_text(
    '# Custom iALS / target ablation + seed robustness\n\n'
    '18 validation evaluations: winner seeds 42/43/44 (10/15 iterations), 3 single-change '
    'target ablations seed42 (10/15), and binary iALS seeds42/43/44 (10/15). '
    'Same fixed VK-LSVD 3000×2000 catalog, rel=0/1/2 evaluator, train weeks 0–24, '
    'holdout week 25. Test closed. No best-seed cherry-picking. '
    'All unknown pairs target=0 confidence=1. '
    'Paired bootstrap describes selected validation users; it is not independent confirmation.\n',
    encoding='utf-8')
print('SUCCESS. Review bundle:',REVIEW)
print('Результаты сохранены.')