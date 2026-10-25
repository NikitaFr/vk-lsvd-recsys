"""ials experiment on train/validation."""
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
OUT = ROOT / "results" / "ials" / "validation"

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

FACTORS=(16, 32, 64)
ALPHAS=(10.0, 40.0)
REGULARIZATIONS=(1.0, 10.0)
CONFIDENCE_MODES=("binary", "graded")
CHECKPOINTS=(5, 10, 15)
SEED=42
BOOTSTRAP_N=2000
BOOTSTRAP_SEED=42
TOP_K=10
# Изменение этого списка является новым экспериментом, не менять задним числом.
FULL_GRID_EXPECTED=24


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
require(len(FACTORS)*len(ALPHAS)*len(REGULARIZATIONS)*len(CONFIDENCE_MODES)==24,
        "Сетка должна содержать 24 уникальных траектории")
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


benchmark_start=time.perf_counter()
bench=IALS(factors=32,alpha=40.,regularization=10.,confidence_mode='binary',random_seed=SEED)
bench.fit(HISTORY, USER_IDS, ITEM_IDS, n_iters=1, history_start_week=0,history_end_week=24)
benchmark_seconds=time.perf_counter()-benchmark_start
require(len(bench.training_log)==1 and np.isfinite(bench.training_log[0]['loss']),
        "Ошибка предварительного запуска iALS")
print(f"Предварительная оценка: f=32 / 1 итерация — {benchmark_seconds:.2f} сек.")
print("Это НЕ точная оценка всех 24 траекторий: f=64 заметно тяжелее, BLAS влияет на скорость.")
print("В случае длительного полного прогона каждая законченная конфигурация будет сохранена на диск.")
del bench
gc.collect()


from itertools import product

CACHE = OUT / "checkpoint_cache"
REVIEW = OUT / "review_bundle"
RUN_FULL_GRID = True  # Зафиксированная полная сетка; False — пропустить обучение после benchmark.
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
run_identity={
    "dataset": "deepvk/VK-LSVD", "revision": REVISION['revision'],
    "files": file_hashes, "modules": module_hashes,
    "ids": {"users": user_hash, "items": item_hash},
    "factors": list(FACTORS), "alphas": list(ALPHAS), "regularizations": list(REGULARIZATIONS),
    "confidence_modes": list(CONFIDENCE_MODES), "checkpoints": list(CHECKPOINTS),
    "seed": SEED, "holdout_week": 25, "top_k": TOP_K,
}
identity_file=OUT/"run_identity.json"
if identity_file.exists():
    prior=json.loads(identity_file.read_text(encoding='utf-8'))
    require(prior == run_identity, "Старый кеш относится к ДРУГОЙ выборке/версии/сетке. "
            "Не подменяй контрольные суммы. Перемести старую папку results/ials/validation целиком "
            "в архив и запусти с чистой папкой.")
else:
    require(not any(CACHE.iterdir()), "Кеш уже существует без run_identity.json: нельзя установить его происхождение")
    identity_file.write_text(json.dumps(run_identity,indent=2,ensure_ascii=False),encoding='utf-8')


def run_id(f, alpha, reg, mode):
    return f"f{int(f)}_alpha{int(alpha)}_lambda{int(reg)}_{mode}"


def stable_checkpoint_csv(cols):
    return pd.DataFrame(cols)

GRID=list(product(FACTORS,ALPHAS,REGULARIZATIONS,CONFIDENCE_MODES))
require(len(GRID)==24 and len(set(GRID))==24, "Некорректное пространство поиска")

if RUN_FULL_GRID:
    start_grid=time.perf_counter()
    for idx,(f,alpha,reg,mode) in enumerate(GRID,1):
        stem=run_id(f,alpha,reg,mode)
        metrics_file=CACHE/f"{stem}_metrics.csv"
        log_file=CACHE/f"{stem}_training_log.csv"
        checkpoints_expected=[(CACHE/f"{stem}_it{t}_per_user.parquet",
                               CACHE/f"{stem}_it{t}_recommendations.parquet") for t in CHECKPOINTS]
        if metrics_file.exists() and log_file.exists() and all(p.exists() for pair in checkpoints_expected for p in pair):
            old_metrics=pd.read_csv(metrics_file)
            require(len(old_metrics)==3 and set(old_metrics.iteration)==set(CHECKPOINTS),
                    f"Кеш {stem} неполон")
            print(f"[{idx:2d}/24] SKIP completed: {stem}")
            continue
        # Incomplete configuration must be recomputed, not spliced with an old trajectory.
        metrics_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)
        for pair in checkpoints_expected:
            for p in pair: p.unlink(missing_ok=True)
        print(f"[{idx:2d}/24] fit: {stem}",flush=True)
        start_fit=time.perf_counter()
        m=IALS(factors=f,alpha=alpha,regularization=reg,
               confidence_mode=mode,random_seed=SEED)
        run_rows=[]
        last_iter=0
        for it,(per_path,rec_path) in zip(CHECKPOINTS,checkpoints_expected):
            if last_iter==0:
                m.fit(HISTORY,USER_IDS,ITEM_IDS,n_iters=it,
                      history_start_week=0,history_end_week=24)
            else:
                m.continue_fit(additional_iterations=it-last_iter)
            last_iter=it
            require(m.n_iterations==it and np.array_equal(m.user_ids,USER_IDS)
                    and np.array_equal(m.item_ids,ITEM_IDS),
                    f"{stem}: неожиданно изменились эмбеддинги/оси")
            scores=m.score()
            require(scores.shape==(3000,2000) and scores.dtype==np.float64
                    and np.isfinite(scores).all(), f"{stem}: некорректная матрица scores")
            ev=evaluate_scores(scores=scores,user_ids=USER_IDS,item_ids=ITEM_IDS,
                               history=HISTORY,holdout=HOLDOUT,k=TOP_K,
                               history_start_week=0,history_end_week=24,holdout_week=25)
            s=ev.summary
            require(s['n_users']==3000 and s['n_eligible']==2082 and
                    s['n_history']==1454216 and s['n_holdout']==16158 and
                    s['holdout_week_checked'] is True and
                    len(ev.per_user)==3000 and len(ev.recommendations)==30000,
                    f"{stem}: оценщик получил неверный набор данных")
            vals={
               'eligible_ndcg_at_10':float(s['eligible']['ndcg_at_10']),
               'eligible_recall_at_10':float(s['eligible']['recall_at_10']),
               'eligible_map_at_10':float(s['eligible']['map_at_10']),
               'all_users_ndcg_at_10':float(s['all_users']['ndcg_at_10']),
               'all_users_recall_at_10':float(s['all_users']['recall_at_10']),
               'all_users_map_at_10':float(s['all_users']['map_at_10']),
            }
            require(all(np.isfinite(v) and 0<=v<=1+1e-12 for v in vals.values()),
                    "Non-finite/out-of-range validation metrics")
            row=dict(factors=f,alpha=alpha,regularization=reg,confidence_mode=mode,
                     iteration=it, **vals,
                     eligible_share=float(s['eligible_share']),
                     n_users=3000,n_eligible=2082,
                     train_loss=float(m.training_log[-1]['loss']),
                     elapsed_seconds=float(time.perf_counter()-start_fit))
            run_rows.append(row)
            # Save complete per-user/recommendation data for independent review and resume.
            ev.per_user.to_parquet(per_path,index=False)
            ev.recommendations.to_parquet(rec_path,index=False)
            print(f"   iter {it:2d}: NDCG={vals['eligible_ndcg_at_10']:.6f}, "
                  f"Recall={vals['eligible_recall_at_10']:.6f}, "
                  f"MAP={vals['eligible_map_at_10']:.6f}, "
                  f"loss={row['train_loss']:.1f}, "
                  f"elapsed={row['elapsed_seconds']:.1f}s",flush=True)
            del scores,ev
        # Diagnostic loss per ALL 15 iterations, not just the 3 checkpoints.
        logs=pd.DataFrame(m.training_log)
        require(len(logs)==15 and (np.diff(logs.loss.to_numpy()) <= 
                1e-8 * np.maximum(1.0, np.abs(logs.loss.to_numpy()[:-1]))).all(),
                f"{stem}: loss неубывает вопреки оптимизации / numerical instability")
        logs.insert(0,'confidence_mode',mode)
        logs.insert(0,'regularization',reg)
        logs.insert(0,'alpha',alpha)
        logs.insert(0,'factors',f)
        tmp_log=log_file.with_suffix('.tmp')
        logs.to_csv(tmp_log,index=False)
        tmp_log.replace(log_file)
        # Completion marker must be written LAST and atomically.
        tmp=metrics_file.with_suffix('.tmp')
        pd.DataFrame(run_rows).to_csv(tmp,index=False)
        tmp.replace(metrics_file)
        del m
        gc.collect()
    print(f"Full grid completed/cached in {time.perf_counter()-start_grid:.1f} s")
else:
    print("RUN_FULL_GRID=False: полноценный поиск не выполнен; не утверждать выбор лучшей модели.")


if RUN_FULL_GRID:
    parts=[]
    for f,alpha,reg,mode in GRID:
        file=CACHE/f"{run_id(f,alpha,reg,mode)}_metrics.csv"
        require(file.exists(),f"Отсутствует завершённая конфигурация {file.name}")
        part=pd.read_csv(file)
        require(len(part)==3 and set(part.iteration)==set(CHECKPOINTS),
                f"Неверные checkpoints: {file.name}")
        parts.append(part)
    RESULTS=pd.concat(parts,ignore_index=True)
    TRAINING_LOGS=pd.concat([pd.read_csv(CACHE/f"{run_id(f,a,r,m)}_training_log.csv")
                             for f,a,r,m in GRID],ignore_index=True)
    require(len(TRAINING_LOGS)==24*15, "Ожидалось 15 значений objective для каждой траектории")
    dims=['factors','alpha','regularization','confidence_mode','iteration']
    require(len(RESULTS)==72 and not RESULTS.duplicated(dims).any(),
            "Ожидалось строго 72 уникальных validation-результата")
    require((RESULTS.n_users==3000).all() and (RESULTS.n_eligible==2082).all()
            and np.allclose(RESULTS.eligible_share,2082/3000),
            "Разные оценки построены на разных наборах пользователей")

    def sort_winners(df):
        return df.sort_values(['eligible_ndcg_at_10','iteration','factors','alpha',
                               'regularization','confidence_mode'],
                              ascending=[False,True,True,True,False,True],
                              kind='mergesort')

    BEST_PER_MODE=sort_winners(RESULTS).drop_duplicates('confidence_mode').sort_values('confidence_mode')
    BEST_OVERALL=sort_winners(RESULTS).iloc[0]
    require(len(BEST_PER_MODE)==2 and BEST_OVERALL.eligible_ndcg_at_10==RESULTS.eligible_ndcg_at_10.max(),
            "Неверный выбор по validation NDCG")
    display(BEST_PER_MODE[dims+['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']])
    print('Best overall:',BEST_OVERALL[dims+['eligible_ndcg_at_10']].to_dict())
    display(sort_winners(RESULTS).head(12)[dims+['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']])


if RUN_FULL_GRID:
    best_by_factor=sort_winners(RESULTS).drop_duplicates(['confidence_mode','factors'])
    plot_figs={}
    for metric, title in [('eligible_ndcg_at_10','NDCG@10'),
                          ('eligible_recall_at_10','Recall@10'),
                          ('eligible_map_at_10','MAP@10')]:
        fig=px.line(best_by_factor.sort_values('factors'),x='factors',y=metric,
                    color='confidence_mode',markers=True,
                    title=f'Best validation {title} by factors (selection: NDCG@10)',
                    labels={'factors':'Латентная размерность f','confidence_mode':'Confidence',metric:title},
                    hover_data=['alpha','regularization','iteration'])
        fig.update_xaxes(type='category')
        fig.show()
        plot_figs[metric]=fig
    trajectories=[]
    for _,w in BEST_PER_MODE.iterrows():
        selected=RESULTS.loc[(RESULTS.factors==w.factors)&(RESULTS.alpha==w.alpha)&
                             (RESULTS.regularization==w.regularization)&
                             (RESULTS.confidence_mode==w.confidence_mode)]
        trajectories.append(selected)
    curves=pd.concat(trajectories)
    fig_it=px.line(curves.sort_values('iteration'),x='iteration',y='eligible_ndcg_at_10',
                   color='confidence_mode',markers=True,
                   title='Validation NDCG@10 по итерациям для лучших двух траекторий',
                   hover_data=['factors','alpha','regularization','train_loss'])
    fig_it.show()
    plot_figs['ndcg_iterations']=fig_it


if RUN_FULL_GRID:
    def ck_paths(row):
        stem=run_id(int(row.factors),float(row.alpha),float(row.regularization),row.confidence_mode)
        it=int(row.iteration)
        return (CACHE/f'{stem}_it{it}_per_user.parquet',
                CACHE/f'{stem}_it{it}_recommendations.parquet')

    per_user_winners={}; recs_winners={}
    for _,w in BEST_PER_MODE.iterrows():
        pu,rc=ck_paths(w)
        per_user_winners[w.confidence_mode]=pd.read_parquet(pu)
        recs_winners[w.confidence_mode]=pd.read_parquet(rc)
        require(len(per_user_winners[w.confidence_mode])==3000 and len(recs_winners[w.confidence_mode])==30000,
                'Неполные сохранённые результаты победителя')
        require(np.array_equal(per_user_winners[w.confidence_mode].user_id.to_numpy(), USER_IDS),
                'В кеше изменился порядок пользователей')
        require(int(per_user_winners[w.confidence_mode].eligible.sum())==2082,
                'В кеше иная eligible-маска')
        require(not pd.MultiIndex.from_frame(recs_winners[w.confidence_mode][['user_id','item_id']]).isin(
                pd.MultiIndex.from_frame(HISTORY[['user_id','item_id']])).any(),
                'В сохранённых рекомендациях есть просмотренные видео')
    ci_rows=[]
    for mode,pu in per_user_winners.items():
        for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
            for group in ('eligible','all_users'):
                ci=bootstrap_mean(pu,metric=metric,group=group,
                                  n_bootstrap=BOOTSTRAP_N,ci=0.95,seed=BOOTSTRAP_SEED)
                ci_rows.append({'confidence_mode':mode,'metric':('map_at_10' if metric=='ap_at_10' else metric),
                                'group':group,**ci})
    BOOTSTRAP=pd.DataFrame(ci_rows)
    paired_rows=[]
    for metric in ('ndcg_at_10','recall_at_10','ap_at_10'):
        for group in ('eligible','all_users'):
            d=bootstrap_paired_difference(per_user_winners['graded'],per_user_winners['binary'],
                                           metric=metric,group=group,
                                           n_bootstrap=BOOTSTRAP_N,ci=0.95,seed=BOOTSTRAP_SEED)
            paired_rows.append({'metric':('map_at_10' if metric=='ap_at_10' else metric),
                                'group':group,**d})
    PAIRED=pd.DataFrame(paired_rows)
    display(BOOTSTRAP[['confidence_mode','metric','group','estimate','ci_low','ci_high']])
    print('graded − binary, парные интервалы:')
    display(PAIRED[['metric','group','estimate','ci_low','ci_high']])


if RUN_FULL_GRID:
    REVIEW.mkdir(parents=True,exist_ok=True)
    RESULTS.sort_values(['confidence_mode','factors','alpha','regularization','iteration']).to_csv(
        REVIEW/'all_72_validation_metrics.csv',index=False)
    TRAINING_LOGS.to_csv(REVIEW/'training_loss_all_iterations.csv',index=False)
    BEST_PER_MODE.to_csv(REVIEW/'best_binary_and_graded.csv',index=False)
    BOOTSTRAP.to_csv(REVIEW/'bootstrap_best_modes.csv',index=False)
    PAIRED.to_csv(REVIEW/'paired_graded_minus_binary.csv',index=False)
    label_table.to_csv(REVIEW/'relevance_counts.csv',index=False)
    for mode in CONFIDENCE_MODES:
        per_user_winners[mode].to_csv(REVIEW/f'winner_{mode}_per_user.csv',index=False)
        recs_winners[mode].to_csv(REVIEW/f'winner_{mode}_recommendations.csv',index=False)
    for metric,fig in plot_figs.items():
        fig.write_html(str(REVIEW/f'{metric}.html'),include_plotlyjs=True,full_html=True)

    selected={
        'selection_metric':'Eligible NDCG@10 (linear gain 0/1/2), validation week 25',
        'tie_break':'higher NDCG; fewer iterations; fewer factors; smaller alpha; larger lambda; binary first',
        'best_overall':BEST_OVERALL.to_dict(),
        'best_per_mode':BEST_PER_MODE.to_dict(orient='records'),
    }
    manifest={
        'status':'PASS_VALIDATION_ONLY',
        'dataset':'deepvk/VK-LSVD','source_revision':REVISION['revision'],
        'input_file_sha256':file_hashes, 'module_sha256':module_hashes,
        'user_ids_sha256':user_hash, 'item_ids_sha256':item_hash,
        'n_train':len(HISTORY),'n_validation':len(HOLDOUT),
        'n_users':len(USER_IDS),'n_items':len(ITEM_IDS),'n_eligible_validation':2082,
        'train_weeks':'0-24','validation_week':25,
        'configurations_fit':24, 'validation_checkpoints_evaluated':72,
        'bootstrap_resamples':BOOTSTRAP_N,'bootstrap_seed':BOOTSTRAP_SEED,
        'selection':selected,'preflight_one_iteration_seconds':benchmark_seconds,
        'test_data_read':False,'test_metrics_calculated':False,
        'train_ranking_metrics_calculated':False,
        'validation_CI_selection_bias_warning':True,
    }
    (REVIEW/'selection_and_manifest.json').write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (REVIEW/'README_results.md').write_text(
        '# VK-LSVD — iALS validation results\n\n'
        'Training 0–24; validation 25; 3,000 original users and 2,000 original items. '
        '72 evaluations from 24 sequential ALS training runs. '
        'Only Eligible NDCG@10 selects configurations. Recall and MAP are diagnostic. '
        'Test feedback was not opened. Bootstrap post-selection is descriptive. '
        'CSV columns user_id/item_id may contain uint64 > 2**63; read using uint64, not float. '
        'The checkpoint_cache folder is kept locally for restart and is not necessary for review.\n',
        encoding='utf-8')
    require(len(list(REVIEW.glob('winner_*_per_user.csv')))==2 and
            len(list(REVIEW.glob('winner_*_recommendations.csv')))==2,
            'Не все выходные файлы победителей записаны')
    print('PASS: полная сетка, графики и CI сохранены в:',REVIEW)
    print('Результаты сохранены.')
    print('Неделя 26 не открывалась; полный refit на train+validation пока не выполнен.')
