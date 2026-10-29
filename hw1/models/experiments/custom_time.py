"""custom time experiment on train/validation."""
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
import shutil
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
OUT = ROOT / "results" / "ials" / "custom_target_time_only_seed_duel"

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

FACTORS = 12
ALPHA = 40.0
REGULARIZATION = 10.0
SEEDS = (42,43,44)
CHECKPOINTS = (10,15)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
TOP_K = 10
FULL_PARAMS = {"q_min":0.4,"watch_weight":0.3,"delta":0.0,"tau":60,"gamma":0.1}
TIME_PARAMS = {**FULL_PARAMS,"watch_weight":0.0}
EXPERIMENTS = {"full":FULL_PARAMS,"time_only":TIME_PARAMS}
RUNS = [(v,seed) for v in EXPERIMENTS for seed in SEEDS]
assert len(RUNS)==6 and len(set(RUNS))==6


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
require(len(RUNS)==6 and CHECKPOINTS==(10,15) and SEEDS==(42,43,44), 'Wrong preregistered comparison')
require(FULL_PARAMS == {"q_min":.4,"watch_weight":.3,"delta":.0,"tau":60,"gamma":.1}, 'Original full target changed')
require(TIME_PARAMS == {"q_min":.4,"watch_weight":.0,"delta":.0,"tau":60,"gamma":.1}, 'Time-only target changed')
require(FACTORS==12 and ALPHA==40. and REGULARIZATION==10., 'IALS changed')
print('PASS: exact project hashes, fixed ID catalog, evaluator and iALS implementation')


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


OUT.mkdir(parents=True,exist_ok=True)
CACHE = OUT/'checkpoint_cache'
REVIEW = OUT/'review_bundle'
CACHE.mkdir(parents=True,exist_ok=True)
RUN_IDENTITY = {
 'experiment':'VK_LSVD_custom_ials_time_only_vs_full_seed_duel_v1.0',
 'file_sha256':file_hashes,'module_sha256':module_hashes,
 'id_sha256':{'users':user_hash,'items':item_hash},
 'targets':EXPERIMENTS,
 'runs':[{'variant':v,'seed':int(seed)} for v,seed in RUNS],
 'ials':{'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
         'confidence_mode':'binary','iterations':list(CHECKPOINTS)},
 'protocol':{'train':'0..24','validation':25,'k':TOP_K,'gains':'linear_0_1_2','test_read':False}
}
ident=OUT/'run_identity.json'
if ident.exists():
    require(json.loads(ident.read_text(encoding='utf-8'))==RUN_IDENTITY,
            'Run identity mismatch — do not reuse this directory')
else:
    require(not any(CACHE.iterdir()),'Cache exists without run identity')
    ident.write_text(json.dumps(RUN_IDENTITY,indent=2,ensure_ascii=False),encoding='utf-8')

def stem_for(variant,seed):
    require(variant in EXPERIMENTS and int(seed) in SEEDS,'Invalid variant/seed')
    return f'{variant}_seed{seed}'

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
print('Runs',len(RUNS),'× checkpoints',CHECKPOINTS,'=',len(RUNS)*len(CHECKPOINTS),'evaluations')


PRIOR = ROOT/'results'/'ials'/'custom_target_diagnostics'
PRIOR_CACHE = PRIOR/'checkpoint_cache'
PRIOR_ID = PRIOR/'run_identity.json'
IMPORTED=[]
EXPECTED_OLD_SEED42_NDCG = {
 ('full',10):0.0269863571, ('full',15):0.0271237941,
 ('time_only',10):0.0282649004, ('time_only',15):0.0287398122
}
# Cache from previous run only contains outputs, no model factors: safely COPY and
# independently validate its numbers and provenance, otherwise train from scratch.
if PRIOR_ID.is_file():
    prior_identity=json.loads(PRIOR_ID.read_text(encoding='utf-8'))
    require(prior_identity.get('experiment')=='VK_LSVD_ials_custom_target_diagnostics_v1.0',
            'Unexpected previous experiment identity')
    require(prior_identity.get('file_sha256')==file_hashes and
            prior_identity.get('module_sha256')==module_hashes and
            prior_identity.get('id_sha256')=={'users':user_hash,'items':item_hash},
            'Historical cache comes from different files or evaluator')
    require(prior_identity.get('ials')=={'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
                                     'confidence_mode':'binary','iterations':list(CHECKPOINTS)},
            'Historical iALS parameters mismatch')
    require(prior_identity['winner_params']==FULL_PARAMS and
            prior_identity['experiments']['time_only']==TIME_PARAMS,
            'Historical target definition mismatch')
    for variant,seed in RUNS:
        if variant=='time_only' and seed!=42:
            continue # These are the NEW runs.
        source_stem=('winner' if variant=='full' else 'time_only')+f'_seed{seed}'
        target_stem=stem_for(variant,seed)
        metrics_file=PRIOR_CACHE/(source_stem+'_metrics.csv')
        log_file=PRIOR_CACHE/(source_stem+'_train_loss.csv')
        paths=[(metrics_file,CACHE/(target_stem+'_metrics.csv')),
               (log_file,CACHE/(target_stem+'_train_loss.csv'))]
        paths += [(PRIOR_CACHE/f'{source_stem}_it{it}_{suffix}.parquet',
                   CACHE/f'{target_stem}_it{it}_{suffix}.parquet')
                  for it in CHECKPOINTS for suffix in ('per_user','recommendations')]
        if all(src.is_file() for src,dst in paths):
            historical=pd.read_csv(metrics_file)
            require(len(historical)==2 and set(historical.iteration)==set(CHECKPOINTS) and
                    set(historical.seed)=={seed} and
                    set(historical.variant)=={('winner' if variant=='full' else 'time_only')},
                    f'Invalid source cached metrics: {source_stem}')
            if seed==42:
                for _,row in historical.iterrows():
                    require(abs(float(row.eligible_ndcg_at_10)-EXPECTED_OLD_SEED42_NDCG[(variant,int(row.iteration))])<2e-6,
                            f'Historical metrics unexpected: {source_stem} it{row.iteration}')
            for it in CHECKPOINTS:
                per=pd.read_parquet(PRIOR_CACHE/f'{source_stem}_it{it}_per_user.parquet')
                require(len(per)==len(USER_IDS) and not per.user_id.duplicated().any() and
                        np.array_equal(np.sort(per.user_id.to_numpy(dtype=np.uint64)),USER_IDS) and
                        per.eligible.sum()==2082, f'Historical per-user row mismatch: {source_stem}/it{it}')
                ndcg=float(per.loc[per.eligible,'ndcg_at_10'].mean())
                original=float(historical.loc[historical.iteration==it,'eligible_ndcg_at_10'].iloc[0])
                require(abs(ndcg-original)<1e-10,'Historical metric != independent per-user recalculation')
            new_metrics=historical.copy()
            new_metrics['variant']=variant
            # Preserve existing validated runs in the NEW cache. Never overwrite.
            all_dsts_exist=all(dst.is_file() for src,dst in paths)
            if not all_dsts_exist:
                require(not any(dst.exists() for src,dst in paths),
                        f'Partially imported run {target_stem}. Remove new cache or resume from trained complete run.')
                for src,dst in paths[1:]:
                    shutil.copy2(src,dst)
                atomic_csv(new_metrics,paths[0][1]) # completion marker LAST
            IMPORTED.append(target_stem)
print('Reused complete historical trajectories:',IMPORTED or '(none: all six train from scratch)')


require(len(RUNS) == 6 and len(CHECKPOINTS)==2, 'Wrong run count')
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
    q = TARGETS[variant]
    q_pos = q[rel_vec>0]
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
print('PASS: all 6 trajectories completed or securely cached')


frames=[];logs=[]
for variant,seed in RUNS:
    stem=stem_for(variant,seed)
    m=CACHE/(stem+'_metrics.csv')
    l=CACHE/(stem+'_train_loss.csv')
    require(m.is_file() and l.is_file(),f'Incomplete trajectory {stem}')
    frame=pd.read_csv(m)
    require(len(frame)==2 and set(frame.iteration)==set(CHECKPOINTS) and set(frame.variant)=={variant} and set(frame.seed)=={seed},
            f'Invalid run metrics {stem}')
    frames.append(frame)
    hist=pd.read_csv(l)
    require(len(hist)==15 and set(hist.iteration)==set(range(1,16)), f'Bad loss log {stem}')
    hist['run_id']=stem; hist['variant']=variant; hist['seed']=seed
    logs.append(hist)
RESULTS=pd.concat(frames,ignore_index=True).sort_values(['variant','seed','iteration']).reset_index(drop=True)
TRAIN_LOGS=pd.concat(logs,ignore_index=True)
require(len(RESULTS)==12 and not RESULTS.duplicated(['variant','seed','iteration']).any(),
        'Expected 12 distinct validation assessments')
require(np.isfinite(RESULTS[['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10','train_loss']]).all().all(),
        'Non-finite score')
SEED_STATS=(RESULTS.groupby(['variant','iteration'],as_index=False)
    .agg(mean_ndcg=('eligible_ndcg_at_10','mean'),std_ndcg=('eligible_ndcg_at_10','std'),
         mean_recall=('eligible_recall_at_10','mean'),std_recall=('eligible_recall_at_10','std'),
         mean_map=('eligible_map_at_10','mean'),std_map=('eligible_map_at_10','std'),
         min_ndcg=('eligible_ndcg_at_10','min'),max_ndcg=('eligible_ndcg_at_10','max'),
         n_seeds=('seed','nunique')))
require((SEED_STATS.n_seeds==3).all(),'Need 3 seeds per cell')
print('PER RUN');display(RESULTS[['variant','seed','iteration','eligible_ndcg_at_10',
                                 'eligible_recall_at_10','eligible_map_at_10']].round(6))
print('MEAN ± STD over exactly 3 declared seeds');display(SEED_STATS.round(6))
wide=RESULTS.pivot(index=['seed','iteration'],columns='variant',values='eligible_ndcg_at_10')
wide['time_minus_full']=wide['time_only']-wide['full']
print('MATCHED-SEED NDCG DELTAS');display(wide.round(6))
for var in EXPERIMENTS:
    expected=EXPECTED_OLD_SEED42_NDCG[(var,10)]
    value=RESULTS.loc[(RESULTS.variant==var)&(RESULTS.seed==42)&(RESULTS.iteration==10),'eligible_ndcg_at_10'].iloc[0]
    require(abs(value-expected)<0.000020, f'Seed42 {var} / it10 does not reproduce prior experiment')
print('PASS: seed42 original checkpoints reproduced')


PAIRED_ROWS=[];AVERAGE_SEED_PAIRED=[]
METRICS=('ndcg_at_10','recall_at_10','ap_at_10')
for it in CHECKPOINTS:
    for metric in METRICS:
        aligned_diffs=[];master_ids=None
        for seed in SEEDS:
            A=pd.read_parquet(per_path(stem_for('time_only',seed),it))
            B=pd.read_parquet(per_path(stem_for('full',seed),it))
            paired=bootstrap_paired_difference(A,B,metric=metric,group='eligible',
                                               n_bootstrap=BOOTSTRAP_N,seed=BOOTSTRAP_SEED)
            PAIRED_ROWS.append({'comparison':'time_only_minus_full','seed':seed,'iteration':it,
                                'metric':metric,**{('bootstrap_seed' if k=='seed' else k):v for k,v in paired.items()}})
            a=A.sort_values('user_id',kind='stable').reset_index(drop=True)
            b=B.sort_values('user_id',kind='stable').reset_index(drop=True)
            require(a.user_id.equals(b.user_id) and a.eligible.equals(b.eligible) and
                    a.n_relevant.equals(b.n_relevant),'Matched users / eligible mismatch')
            if master_ids is None: master_ids=a.user_id
            else: require(master_ids.equals(a.user_id),'Users differ between seeds')
            eligible=a.eligible.to_numpy(dtype=bool)
            d=a.loc[eligible,metric].to_numpy(dtype=float)-b.loc[eligible,metric].to_numpy(dtype=float)
            require(len(d)==2082 and np.isfinite(d).all(),'Malformed eligible differences')
            aligned_diffs.append(d)
        # Resample USER IDs ONCE, after averaging the three paired-seed differences
        # for each user. Same users appear in all seeds and are not independent.
        average_d=np.mean(np.stack(aligned_diffs),axis=0)
        rng=np.random.default_rng(BOOTSTRAP_SEED)
        means=np.empty(BOOTSTRAP_N,dtype=float)
        for b_idx in range(BOOTSTRAP_N):
            means[b_idx]=np.mean(average_d[rng.integers(0,len(average_d),len(average_d))])
        lo,hi=np.quantile(means,[0.025,0.975])
        AVERAGE_SEED_PAIRED.append({'iteration':it,'metric':metric,'estimate':average_d.mean(),
            'ci_low':lo,'ci_high':hi,'n_users':len(average_d),
            'n_seeds':3,'n_bootstrap':BOOTSTRAP_N,
            'method':'bootstrap eligible users after averaging paired differences over 3 fixed seeds'})
PAIRED=pd.DataFrame(PAIRED_ROWS)
AVERAGE_PAIRED=pd.DataFrame(AVERAGE_SEED_PAIRED)
require(len(PAIRED)==18 and len(AVERAGE_PAIRED)==6,'Wrong comparison count')
print('PER-SEED NDCG DIFFERENCES (time_only – full):')
display(PAIRED.loc[PAIRED.metric=='ndcg_at_10',['seed','iteration','estimate','ci_low','ci_high']].round(6))
print('AVERAGE-SEED NDCG DIFFERENCES (paired user bootstrap, same users):')
display(AVERAGE_PAIRED.loc[AVERAGE_PAIRED.metric=='ndcg_at_10'].round(6))
print('Important: these CIs do not account for model selection on the same validation or untested seeds.')


# Main rule declared before NEW seed-43/44 results:
# maximize 3-seed mean Eligible NDCG@10 across 2 target formulas × 2 iterations;
# report selection uncertainty, do not cherry-pick seed.
RANKED=SEED_STATS.sort_values(['mean_ndcg','variant','iteration'],ascending=[False,True,True]).reset_index(drop=True)
CHOSEN=RANKED.iloc[0]
best_variant=str(CHOSEN.variant);best_iteration=int(CHOSEN.iteration)
print('PROVISIONAL WINNER by predeclared mean NDCG: ',best_variant, best_iteration,
      'iterations, mean=',f'{CHOSEN.mean_ndcg:.6f}')
print('FIXED iALS:',f'factors={FACTORS}, alpha={ALPHA:g}, regularization={REGULARIZATION:g}, confidence_mode=binary')
print('FIXED custom target:',EXPERIMENTS[best_variant])
print('IMPORTANT: before final freeze, compare paired uncertainty and variation across seeds.')
display(RANKED[['variant','iteration','mean_ndcg','std_ndcg','mean_recall','mean_map']].round(6))


FIGURES={}
fig=px.line(RESULTS,x='iteration',y='eligible_ndcg_at_10',color='variant',line_dash=RESULTS.seed.astype(str),
            markers=True,title='NDCG@10 по seeds 42/43/44',
            labels={'iteration':'Итерации','eligible_ndcg_at_10':'Eligible NDCG@10','variant':'Target'})
fig.update_layout(height=520);fig.show();FIGURES['ndcg_seeds']=fig
fig=px.line(SEED_STATS,x='iteration',y='mean_ndcg',color='variant',error_y='std_ndcg',markers=True,
            title='Средняя NDCG@10 ± std по 3 seeds',
            labels={'iteration':'Итерации','mean_ndcg':'Mean Eligible NDCG@10'})
fig.update_layout(height=510);fig.show();FIGURES['ndcg_mean_std']=fig
fig=px.bar(wide.reset_index(),x='seed',y='time_minus_full',color=wide.reset_index().iteration.astype(str),barmode='group',
            title='Парное преимущество Time only − Full по seeds (положительное = Time only лучше)',
            labels={'time_minus_full':'Δ NDCG@10','seed':'Seed','color':'Iterations'})
fig.update_layout(height=470);fig.show();FIGURES['ndcg_paired_differences']=fig
fig=px.line(TRAIN_LOGS.loc[TRAIN_LOGS.seed==42],x='iteration',y='loss',color='variant',
            title='Train ALS objective, seed=42 (не метрика качества рекомендаций)')
fig.update_layout(height=490);fig.show();FIGURES['training_objective']=fig


REVIEW.mkdir(parents=True,exist_ok=True)
RESULTS.to_csv(REVIEW/'all_12_validation_metrics.csv',index=False)
SEED_STATS.to_csv(REVIEW/'seed_mean_std.csv',index=False)
wide.reset_index().to_csv(REVIEW/'paired_ndcg_by_seed.csv',index=False)
PAIRED.to_csv(REVIEW/'paired_bootstrap_by_seed.csv',index=False)
AVERAGE_PAIRED.to_csv(REVIEW/'paired_bootstrap_mean_of_seeds.csv',index=False)
TRAIN_LOGS.to_csv(REVIEW/'all_training_iterations.csv',index=False)
label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
for variant in EXPERIMENTS:
    for seed in SEEDS:
        for it in CHECKPOINTS:
            stem=stem_for(variant,seed)
            data=pd.read_parquet(per_path(stem,it))
            require(len(data)==len(USER_IDS),'Bad final per-user output')
            data.to_csv(REVIEW/f'{stem}_it{it}_per_user.csv',index=False)
for variant in EXPERIMENTS:
    stem=stem_for(variant,42)
    pd.read_parquet(rec_path(stem,15)).to_csv(REVIEW/f'{stem}_it15_recommendations.csv',index=False)
for name,fig in FIGURES.items():
    fig.write_html(str(REVIEW/(name+'.html')),include_plotlyjs=True,full_html=True)
rel_vec=TRAIN_SIGNALS.rel.to_numpy()
dist=[]
for name,q in TARGETS.items():
    for rel in (0,1,2):
        sub=q[rel_vec==rel]
        dist.append({'variant':name,'rel':rel,'count':len(sub),'mean':sub.mean(),
                     'p10':np.quantile(sub,.1),'median':np.median(sub),'p90':np.quantile(sub,.9),'max':sub.max()})
pd.DataFrame(dist).to_csv(REVIEW/'target_distribution_by_rel.csv',index=False)
selection={'criterion':'max mean Eligible NDCG@10 across 3 fixed seeds, two target formulas, checkpoints 10 or 15',
           'chosen_variant':best_variant,'chosen_iterations':best_iteration,
           'seed_mean_ndcg':float(CHOSEN.mean_ndcg),'seed_std_ndcg':float(CHOSEN.std_ndcg),
           'custom_target_params':EXPERIMENTS[best_variant],
           'ials':{'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
                   'confidence_mode':'binary','seeds':list(SEEDS)},
           'caveat':'validation selection only; no test used; compare paired CIs before deciding final freeze'}
(REVIEW/'provisional_selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False),encoding='utf-8')
manifest={'experiment':'VK-LSVD Custom iALS final seed comparison v1.0',
          'file_sha256':file_hashes,'module_sha256':module_hashes,
          'ids_sha256':{'users':user_hash,'items':item_hash},
          'dataset':{'train_weeks':'0..24','validation_week':25,'test_read':False,
                     'train_rows':int(len(HISTORY)),'validation_rows':int(len(HOLDOUT)),
                     'users':len(USER_IDS),'items':len(ITEM_IDS)},
          'ials':{'factors':FACTORS,'alpha':ALPHA,'regularization':REGULARIZATION,
                  'confidence_mode':'binary','seeds':list(SEEDS),'iterations':list(CHECKPOINTS)},
          'target_params':EXPERIMENTS,'historical_cache_imported':IMPORTED,
          'metric':'Eligible NDCG@10, linear gains, remove all train seen items',
          'unknown_pairs':'preference=0 confidence=1',
          'bootstrap':'per-user percentile; same user ids shared across seeds',
          'prior_validation_tuning':True,'not_choose_best_seed_on_test':True}
(REVIEW/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(REVIEW/'README_results.md').write_text(
 '# Final Time only vs Full continuous — Custom iALS\n\n'
 '6 trajectories × checkpoints 10/15 = 12 assessments, seeds 42,43,44, same validation users.\n'
 'Included complete per-user CSV, seed-level and across-seed paired bootstrap, aggregate mean/std, metrics and plots.\n'
 'Historical cache is copied only after strict hash and metric verification; missing runs trained from scratch.\n'
 'The preliminary test choice is recorded in provisional_selection.json. It is not an independent evaluation.\n'
 'The test data was never loaded. Do not tune on test.\n',encoding='utf-8')
print('SUCCESS: 12 evaluations / 6 models, review bundle ->',REVIEW)
print('Archive this REVIEW folder and send it back; do not include raw data / closed test.')
