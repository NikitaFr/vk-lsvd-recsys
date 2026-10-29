"""custom ials experiment on train/validation."""
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
OUT = ROOT / "results" / "ials" / "custom_target_validation"

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

# Новый эксперимент: фиксированы параметры iALS, варьируется ТОЛЬКО target.
FACTORS = 12
ALPHA = 40.0
REGULARIZATION = 10.0
SEED = 42
GAMMA = 0.1
CHECKPOINTS = (5, 10)
Q_MINS = (0.4, 0.7)
WATCH_WEIGHTS = (0.3, 0.6, 0.8)
REWATCH_DELTAS = (0.0, 0.2)
TIME_TAUS = (30, 60)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
TOP_K = 10
EXPECTED_GRID_SIZE = 24


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
require(len(Q_MINS)*len(WATCH_WEIGHTS)*len(REWATCH_DELTAS)*len(TIME_TAUS)==24, "Некорректная сетка target")
require(CHECKPOINTS==(5,10) and SEED==42 and FACTORS==12 and ALPHA==40. and REGULARIZATION==10., "Базовые параметры iALS изменены")
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


GRID=list(product(Q_MINS,WATCH_WEIGHTS,REWATCH_DELTAS,TIME_TAUS))
require(len(GRID)==EXPECTED_GRID_SIZE and len(set(GRID))==EXPECTED_GRID_SIZE, 'Duplicate config')
OUT.mkdir(parents=True,exist_ok=True)
CACHE=OUT/'checkpoint_cache'
REVIEW=OUT/'review_bundle'
CACHE.mkdir(parents=True,exist_ok=True)
RUN_IDENTITY={
 'experiment':'VK_LSVD_iALS_custom_continuous_v1.0',
 'formula':'rel0_to_zero_positive_min1_qmin_plus_mix_frac_logtime_gamma_clip1',
 'data_sha256':file_hashes,'module_sha256':module_hashes,
 'ids_sha256':{'users':user_hash,'items':item_hash},
 'grid':{'q_min':Q_MINS,'w':WATCH_WEIGHTS,'delta':REWATCH_DELTAS,'tau':TIME_TAUS,'gamma':GAMMA},
 'ials':{'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,'seed':SEED,
         'confidence_mode':'binary','checkpoints':CHECKPOINTS},
 'evaluation':{'k':10,'train_weeks':'0..24','validation_week':25,'gains':'linear 0/1/2'} }
identity=OUT/'run_identity.json'
if identity.exists():
    require(json.loads(identity.read_text(encoding='utf-8')) == json.loads(json.dumps(RUN_IDENTITY)),
            'Cache mismatch: another experiment or modified files; rename old folder')
else:
    require(not any(CACHE.iterdir()), 'Cache exists but no run identity file')
    identity.write_text(json.dumps(RUN_IDENTITY,indent=2,ensure_ascii=False),encoding='utf-8')

def config_id(q_min, w, delta, tau):
    def tag(x): return str(x).replace('.','p')
    return f'q{tag(q_min)}_w{tag(w)}_d{tag(delta)}_tau{int(tau)}'

def cache_paths(stem,it):
    return (CACHE/f'{stem}_it{it}_per_user.parquet',CACHE/f'{stem}_it{it}_recommendations.parquet')

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
print('Grid size',len(GRID),'× checkpoints',CHECKPOINTS,'=',len(GRID)*len(CHECKPOINTS),'assessments')


for i,(q_min,w,delta,tau) in enumerate(GRID,1):
    stem=config_id(q_min,w,delta,tau)
    done=CACHE/f'{stem}_metrics.csv'
    logfile=CACHE/f'{stem}_train_loss.csv'
    cp=[cache_paths(stem,it) for it in CHECKPOINTS]
    if done.is_file() and logfile.is_file() and all(p.is_file() for pair in cp for p in pair):
        old=pd.read_csv(done)
        require(len(old)==2 and set(old.iteration)==set(CHECKPOINTS),f'Broken cached run {stem}')
        print(f'[{i:02d}/{len(GRID)}] CACHED {stem}')
        continue
    # Partial trajectory cannot resume from weights: start it over, leave finished others intact.
    done.unlink(missing_ok=True); logfile.unlink(missing_ok=True)
    for pair in cp:
        for p in pair: p.unlink(missing_ok=True)
    q=continuous_target(TRAIN_SIGNALS,q_min=q_min,watch_weight=w,delta=delta,tau=tau,gamma=GAMMA)
    positive=q[TRAIN_SIGNALS.rel.to_numpy()>0]
    q_stats={'target_mean_positive':float(positive.mean()),'target_median_positive':float(np.median(positive)),
             'target_p10_positive':float(np.quantile(positive,.1)),
             'target_p90_positive':float(np.quantile(positive,.9)),
             'target_saturated_positive':float(np.mean(positive>=1.))}
    m=IALS(factors=FACTORS,alpha=ALPHA,regularization=REGULARIZATION,
           confidence_mode='binary',random_seed=SEED)
    start=time.perf_counter()
    rows=[]; completed=0
    print(f'[{i:02d}/{len(GRID)}] TRAIN {stem}',flush=True)
    for it in CHECKPOINTS:
        if not completed:
            m.fit(HISTORY,USER_IDS,ITEM_IDS,n_iters=it,observed_targets=q,
                  history_start_week=0,history_end_week=24)
        else:
            m.continue_fit(additional_iterations=it-completed)
        completed=it
        ev=evaluate_model(m)
        row={'variant':'continuous','run_id':stem,'q_min':q_min,'w':w,'delta':delta,'tau':tau,
             'gamma':GAMMA,'iteration':it,**q_stats,**metrics_from_eval(ev),
             'train_loss':float(m.training_log[-1]['loss']),
             'elapsed_seconds':float(time.perf_counter()-start)}
        rows.append(row)
        a,b=cache_paths(stem,it)
        ev.per_user.to_parquet(a,index=False)
        ev.recommendations.to_parquet(b,index=False)
        print(f'  iter {it}: NDCG {row["eligible_ndcg_at_10"]:.6f}  '
              f'Recall {row["eligible_recall_at_10"]:.6f}  MAP {row["eligible_map_at_10"]:.6f}',flush=True)
        del ev
    log=pd.DataFrame(m.training_log)
    require(len(log)==max(CHECKPOINTS) and np.isfinite(log.loss).all(),'Bad ALS training log')
    log.insert(0,'run_id',stem)
    log.to_csv(logfile.with_suffix('.tmp'),index=False)
    logfile.with_suffix('.tmp').replace(logfile)
    pd.DataFrame(rows).to_csv(done.with_suffix('.tmp'),index=False)
    done.with_suffix('.tmp').replace(done)  # completion marker LAST
    del m,q,positive
    gc.collect()
print('PASS: every continuous target trajectory complete or cached')


metrics=[]; logs=[]
for q_min,w,delta,tau in GRID:
    stem=config_id(q_min,w,delta,tau)
    f=CACHE/f'{stem}_metrics.csv'
    require(f.is_file(),f'Missing {stem}')
    d=pd.read_csv(f)
    require(len(d)==2 and set(d.iteration)==set(CHECKPOINTS),f'Bad metrics {stem}')
    metrics.append(d)
    logs.append(pd.read_csv(CACHE/f'{stem}_train_loss.csv'))
RESULTS=pd.concat(metrics,ignore_index=True)
TRAIN_LOGS=pd.concat(logs,ignore_index=True)
require(len(RESULTS)==48 and not RESULTS.duplicated(['run_id','iteration']).any(), 'Not 48 unique evaluations')
RESULTS=RESULTS.sort_values(['eligible_ndcg_at_10','iteration','run_id'],ascending=[False,True,True],kind='stable').reset_index(drop=True)
WINNER=RESULTS.iloc[0]
WINNER_STEM=str(WINNER.run_id)
WINNER_IT=int(WINNER.iteration)
print('BEST',WINNER[['run_id','iteration','eligible_ndcg_at_10','eligible_recall_at_10',
                     'eligible_map_at_10','target_mean_positive']].to_dict())
display(RESULTS.head(12)[['run_id','q_min','w','delta','tau','iteration','eligible_ndcg_at_10',
                          'eligible_recall_at_10','eligible_map_at_10']])


# IMPORTANT: baseline is trained afresh on identical data/seed; it does not depend on validation tuning.
# constant-scale baseline is a post-hoc diagnostic keyed to the winning target mean.
controls=[('binary',None),('constant_scale',float(WINNER.target_mean_positive))]
control_rows=[]
for name,c_value in controls:
    stem=f'control_{name}'
    done=CACHE/f'{stem}_metrics.csv'
    cp=CACHE/f'{stem}_per_user.parquet'
    rec=CACHE/f'{stem}_recommendations.parquet'
    if done.exists() and cp.exists() and rec.exists():
        part=pd.read_csv(done)
        require(len(part)==1,'Bad control cache')
        control_rows.append(part.iloc[0].to_dict())
        print('CACHED',stem)
        continue
    target=None if name=='binary' else np.where(TRAIN_SIGNALS.rel.to_numpy()>0,c_value,0.)
    m=IALS(factors=FACTORS,alpha=ALPHA,regularization=REGULARIZATION,
           confidence_mode='binary',random_seed=SEED)
    start=time.perf_counter()
    m.fit(HISTORY,USER_IDS,ITEM_IDS,n_iters=10,observed_targets=target,
          history_start_week=0,history_end_week=24)
    ev=evaluate_model(m)
    vals=metrics_from_eval(ev)
    row={'variant':name,'run_id':stem,'iteration':10,'target_mean_positive':(1. if target is None else c_value),
         'train_loss':float(m.training_log[-1]['loss']),
         'elapsed_seconds':float(time.perf_counter()-start),**vals}
    ev.per_user.to_parquet(cp,index=False)
    ev.recommendations.to_parquet(rec,index=False)
    pd.DataFrame([row]).to_csv(done.with_suffix('.tmp'),index=False)
    done.with_suffix('.tmp').replace(done)
    control_rows.append(row)
    print(name,'NDCG:',vals['eligible_ndcg_at_10'])
    del m,ev
    gc.collect()
CONTROLS=pd.DataFrame(control_rows)
reference=float(CONTROLS.loc[CONTROLS.variant=='binary','eligible_ndcg_at_10'].iloc[0])
require(abs(reference-0.020702)<0.00002,
        'Binary baseline differs from the frozen iALS experiment; stop and diagnose')
print('PASS: unchanged baseline agrees with old validation')


figures={}
fig=px.scatter(RESULTS,x='target_mean_positive',y='eligible_ndcg_at_10',color='q_min',
               symbol='iteration',hover_data=['w','delta','tau','run_id'],
               title='Custom iALS: средний positive target и NDCG@10')
fig.add_hline(y=reference,line_dash='dash',annotation_text='Binary iALS baseline')
fig.update_layout(height=500)
fig.show(); figures['target_scale_vs_ndcg']=fig

for dim in ('w','q_min','tau','delta'):
    agg=RESULTS.loc[RESULTS.iteration==10].groupby(dim,as_index=False)['eligible_ndcg_at_10'].mean()
    f=px.bar(agg,x=dim,y='eligible_ndcg_at_10',title=f'Средняя NDCG@10 по {dim} (10 итераций)')
    f.update_xaxes(type='category'); f.show(); figures[f'by_{dim}']=f

compare=pd.concat([RESULTS[['run_id','variant','iteration','eligible_ndcg_at_10']].head(5),
                   CONTROLS[['run_id','variant','iteration','eligible_ndcg_at_10']]],ignore_index=True)
display(compare)


winning_per=pd.read_parquet(cache_paths(WINNER_STEM,WINNER_IT)[0])
binary_per=pd.read_parquet(CACHE/'control_binary_per_user.parquet')
scale_per=pd.read_parquet(CACHE/'control_constant_scale_per_user.parquet')
require(len(winning_per)==len(binary_per)==len(scale_per)==3000,'Mismatch in per-user tables')
paired=[]; ci=[]
for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
    ci.append({'metric':metric,'model':'winner',**bootstrap_mean(winning_per,metric=metric,group='eligible',
                 n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)})
    for tag,other in [('binary',binary_per),('constant_scale',scale_per)]:
        res=bootstrap_paired_difference(winning_per,other,metric=metric,group='eligible',
                                      n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
        paired.append({'metric':metric,'difference':'winner_minus_'+tag,**res})
PAIRED=pd.DataFrame(paired)
CI=pd.DataFrame(ci)
display(PAIRED[['metric','difference','estimate','ci_low','ci_high']])

wq=continuous_target(TRAIN_SIGNALS,q_min=float(WINNER.q_min),watch_weight=float(WINNER.w),
                     delta=float(WINNER.delta),tau=int(WINNER.tau),gamma=GAMMA)
DIST=pd.DataFrame({'rel':TRAIN_SIGNALS.rel.to_numpy(dtype=np.int8),'q':wq,
                   'duration':TRAIN_SIGNALS.duration.to_numpy(dtype=float)})
DIAGNOSTIC=DIST.groupby('rel').q.agg(['count','mean','median','min','max']).reset_index()
display(DIAGNOSTIC)
# No target attached to validation or evaluator — train only.


REVIEW.mkdir(parents=True,exist_ok=True)
RESULTS.to_csv(REVIEW/'all_48_validation_metrics.csv',index=False)
TRAIN_LOGS.to_csv(REVIEW/'all_training_iterations.csv',index=False)
CONTROLS.to_csv(REVIEW/'controls_binary_and_constant_scale.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_winner_minus_controls.csv',index=False)
CI.to_csv(REVIEW/'bootstrap_winner.csv',index=False)
DIAGNOSTIC.to_csv(REVIEW/'winner_target_by_rel.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
winning_per.to_csv(REVIEW/'winner_per_user.csv',index=False)
pd.read_parquet(cache_paths(WINNER_STEM,WINNER_IT)[1]).to_csv(REVIEW/'winner_recommendations.csv',index=False)
binary_per.to_csv(REVIEW/'control_binary_per_user.csv',index=False)
scale_per.to_csv(REVIEW/'control_constant_scale_per_user.csv',index=False)
for name,fig in figures.items():
    fig.write_html(str(REVIEW/f'{name}.html'),include_plotlyjs=True,full_html=True)
summary={
 'experiment':'VK-LSVD iALS custom continuous target validation 1.0',
 'input_sha256':file_hashes,'module_sha256':module_hashes,
 'target_formula':'see notebook first cell and continuous_target implementation',
 'winner':WINNER.to_dict(),
 'controls':CONTROLS.to_dict(orient='records'),
 'n_target_configurations':24,'n_validation_points':48,
 'week_train':'0..24','week_validation':25,'test_read':False,
 'selected_by':'Eligible NDCG@10, linear gains 0/1/2',
 'post_selection_bootstrap_warning':True,
 'constant_scale_is_posthoc_diagnostic':True,
 'unknown_pairs':'all nonobserved user-item pairs q=0 confidence=1',
 'ials':{'factors':12,'alpha':40,'regularization':10,'seed':42,'iterations':CHECKPOINTS}
}
(REVIEW/'selection_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(REVIEW/'README_results.md').write_text(
    '# VK-LSVD — Custom iALS validation results\n\n'
    'Train=weeks 0..24, holdout=week 25, test closed. '
    '48 validation evaluations, 24 continuous target configs, '
    'binary and constant-scale controls. '
    'Ground truth rel 0/1/2 unchanged, 3000×2000 catalog; '
    'all unobserved pairs handled implicitly by ALS with target=0 confidence=1. '
    'Winner picked by Eligible NDCG@10; bootstrap post-selection only descriptive. '
    'user_id/item_id are uint64, do not convert to float for joins.\n',encoding='utf-8')
print('PASS: all exports saved under',REVIEW)
print('ПРИШЛИ ZIP папки review_bundle. НЕДЕЛЯ 26 НЕ ОТКРЫВАЛАСЬ.')
