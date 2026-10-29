"""item knn experiment on train/validation."""
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
OUT = ROOT / "results" / "item_knn" / "validation"

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
    "models/item_knn.py": "06ab92db064f1c0f793d6d136beef9289bcedcbd6676e095a34707887a728bc4",
    "models/__init__.py": "8d79fbdaf9152739cdc064713068980571c2cc36497b906a3b2b2190f67b5698",
    "models/utils/evaluation/__init__.py": "e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c",
    "models/utils/evaluation/_validation.py": "6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405",
    "models/utils/evaluation/bootstrap.py": "9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221",
    "models/utils/evaluation/metrics.py": "14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938",
    "models/utils/evaluation/pipeline.py": "27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f",
    "models/utils/evaluation/ranking.py": "5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2",
    "models/utils/evaluation/relevance.py": "ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1",
}
K_GRID = (5, 10, 20, 40, 80, 160, 320, 640, 1000, 1500, 1999)
SIGNALS = ("binary", "graded")
AGGREGATIONS = ("sum", "normalized")
METRICS = ("ndcg_at_10", "recall_at_10", "map_at_10")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uint64_ids(values) -> str:
    # Побайтово повторяем исходный split-аудит: sort(uint64) -> bytes -> SHA-256.
    a = np.asarray(values)
    require(a.ndim == 1 and a.dtype.kind in "iu", "ID должны оставаться целочисленными")
    require(not (a < 0).any(), "В датасете EDA не должно быть отрицательных ID")
    return hashlib.sha256(np.sort(a.astype(np.uint64)).tobytes()).hexdigest()

missing = [str(p) for p in FILES.values() if not p.is_file()]
missing.extend(str(ROOT / file) for file in EXPECTED_MODULE_SHA256
               if not (ROOT / file).is_file())
if missing:
    raise FileNotFoundError("В существующем проекте не найдены необходимые файлы: " + repr(missing))
print("ROOT:", ROOT)
print("Python", platform.python_version(), "| pandas", pd.__version__,
      "| numpy", np.__version__, "| scipy", scipy.__version__, "| plotly", plotly.__version__)
print("PASS: пути исходных данных и моделей найдены; файлы недели 26 нигде не загружаются.")


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
from models import ItemKNN
from models.item_knn import K_GRID as MODEL_K_GRID, SIGNALS as MODEL_SIGNALS, AGGREGATIONS as MODEL_AGGREGATIONS
from models.utils.evaluation import label_relevance, evaluate_scores, bootstrap_mean, bootstrap_paired_difference
require(K_GRID == MODEL_K_GRID and SIGNALS == MODEL_SIGNALS and AGGREGATIONS == MODEL_AGGREGATIONS,
        "Сетка или режимы не соответствуют модели, заданной в эксперименте")
print("PASS: SHA-256 Parquet, исходный manifest и все согласованные .py совпали")


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

# Отдаём обеим моделям одинаковый размеченный train; сам ItemKNN выберет signal.
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


# Крошечный тест интерфейса перед длинным экспериментом; не использует будущие данные.
tiny_u = np.array([2**63 + 5, 2**63 + 6], dtype=np.uint64)
tiny_i = np.array([11, 12, 13], dtype=np.uint64)
tiny_h = pd.DataFrame({
    "user_id": [tiny_u[0], tiny_u[0], tiny_u[1]],
    "item_id": [tiny_i[0], tiny_i[1], tiny_i[1]],
    "week": [0, 0, 0], "rel": [2, 1, 1]
})
tiny_model = ItemKNN(signal="graded").fit(tiny_h, tiny_u, tiny_i,
                                            history_start_week=0, history_end_week=24)
tiny_scores = tiny_model.score_both(k=1)
require(set(tiny_scores) == {"sum", "normalized"}, "score_both вернул не две формулы")
require(all(v.shape == (2, 3) and np.isfinite(v).all() for v in tiny_scores.values()),
        "Некорректные scores или потеря широких uint64 ID")
require((tiny_model.similarity.diagonal() == 0).all(), "Диагональ similarity должна быть нулевой")
require(np.array_equal(tiny_model.user_ids, tiny_u), "Модель меняет порядок ID")
print("PASS: API ItemKNN, две формулы и uint64 ID проверены до full fit")
del tiny_model, tiny_scores


# По заранее утверждённой основной метрике; tie-break не использует Recall/MAP.
def selection_key(row):
    return (-float(row["eligible_ndcg_at_10"]),
            int(row["k_neighbors"]),
            0 if row["aggregation"] == "sum" else 1)

all_rows = []
winner_by_group = {}  # (signal, aggregation) -> {row, per_user, recommendations}
fit_stats = []
start_experiment = time.perf_counter()

for signal in SIGNALS:
    print(f"\n--- fit(signal={signal}) on 0–24; validation only week 25 ---")
    model = ItemKNN(signal=signal)
    fit_start = time.perf_counter()
    model.fit(HISTORY, USER_IDS, ITEM_IDS, history_start_week=0, history_end_week=24)
    fit_sec = time.perf_counter() - fit_start
    require(np.array_equal(model.user_ids, USER_IDS) and np.array_equal(model.item_ids, ITEM_IDS),
            "Модель изменила оси исходной выборки")
    sim = model.similarity
    require(sim.shape == (2000, 2000) and np.isfinite(sim).all(), "Неверная similarity")
    require(np.all(sim >= 0) and np.all(sim <= 1 + 1e-12), "Cosine вне [0,1]")
    require(np.all(np.diag(sim) == 0), "Self-similarity должна быть принудительно 0")
    fit_stats.append({"signal": signal, "fit_seconds": fit_sec,
                      "similarity_fraction_nonzero_offdiag": float(np.count_nonzero(sim) / (2000 * 1999)),
                      "similarity_max_offdiag": float(sim.max())})
    del sim
    print(f"fit {signal}: {fit_sec:.2f} s")

    for k_neighbors in K_GRID:
        loop_start = time.perf_counter()
        both = model.score_both(k=k_neighbors)  # вычисляем числитель ОДИН раз для обоих scorings
        require(set(both) == set(AGGREGATIONS), "score_both вернул неизвестные агрегации")
        for aggregation in AGGREGATIONS:
            scores = both[aggregation]
            require(scores.shape == (3000, 2000) and np.isfinite(scores).all(),
                    f"{signal}/{aggregation}/k={k_neighbors}: неверные scores")
            ev = evaluate_scores(
                scores=scores, user_ids=USER_IDS, item_ids=ITEM_IDS,
                history=HISTORY, holdout=HOLDOUT, k=10,
                history_start_week=0, history_end_week=24, holdout_week=25,
            )
            s = ev.summary
            require(s["n_users"] == 3000 and s["n_eligible"] == 2082 and
                    s["n_holdout"] == 16158 and s["n_history"] == 1454216 and
                    s["holdout_week_checked"] is True,
                    f"{signal}/{aggregation}/k={k_neighbors}: evaluator получил не те данные")
            require(len(ev.recommendations) == 30000 and len(ev.per_user) == 3000,
                    "Неверное число рекомендаций/персональных метрик")
            row = {
                "signal": signal,
                "aggregation": aggregation,
                "k_neighbors": int(k_neighbors),
                "top_k": 10,
                "eligible_ndcg_at_10": float(s["eligible"]["ndcg_at_10"]),
                "eligible_recall_at_10": float(s["eligible"]["recall_at_10"]),
                "eligible_map_at_10": float(s["eligible"]["map_at_10"]),
                "all_users_ndcg_at_10": float(s["all_users"]["ndcg_at_10"]),
                "all_users_recall_at_10": float(s["all_users"]["recall_at_10"]),
                "all_users_map_at_10": float(s["all_users"]["map_at_10"]),
                "n_users": int(s["n_users"]),
                "n_eligible": int(s["n_eligible"]),
                "eligible_share": float(s["eligible_share"]),
                "top10_zero_score_fraction": float((ev.recommendations["score"] == 0).mean()),
            }
            require(all(np.isfinite(row[v]) and 0 <= row[v] <= 1 + 1e-12 for v in (
                "eligible_ndcg_at_10", "eligible_recall_at_10", "eligible_map_at_10",
                "all_users_ndcg_at_10", "all_users_recall_at_10", "all_users_map_at_10")),
                "Метрики не конечны или не в [0,1]")
            all_rows.append(row)
            key = (signal, aggregation)
            previous = winner_by_group.get(key)
            if previous is None or selection_key(row) < selection_key(previous["row"]):
                winner_by_group[key] = {
                    "row": row.copy(),
                    "per_user": ev.per_user.copy(deep=True),
                    "recommendations": ev.recommendations.copy(deep=True),
                }
            del ev, scores
        del both
        print(f"  k={k_neighbors:4d}: "
              f"sum NDCG={all_rows[-2]['eligible_ndcg_at_10']:.6f} | "
              f"normalized NDCG={all_rows[-1]['eligible_ndcg_at_10']:.6f} | "
              f"{time.perf_counter()-loop_start:.1f} s")
    del model
    gc.collect()

experiment_seconds = time.perf_counter() - start_experiment
results = pd.DataFrame(all_rows)
require(len(results) == 44 and
        not results.duplicated(["signal", "aggregation", "k_neighbors"]).any(),
        "Эксперимент должен иметь ровно 44 уникальные конфигурации")
require(set(map(tuple, results[["signal", "aggregation"]].drop_duplicates().to_numpy())) ==
        {(s, a) for s in SIGNALS for a in AGGREGATIONS}, "Не все варианты сигнала/скоринга протестированы")
require(len(winner_by_group) == 4, "Должно остаться четыре независимо выбранных победителя")
print("\nPASS: 44/44 configurations completed, train=0–24, validation=25, test not accessed")
print(f"Experiment elapsed time: {experiment_seconds:.1f} s")
display(results.sort_values(["signal", "aggregation", "k_neighbors"]).reset_index(drop=True))


best_four = pd.DataFrame([v["row"] for v in winner_by_group.values()])
best_four = best_four.sort_values(["signal", "aggregation"]).reset_index(drop=True)
best_per_signal = pd.DataFrame([
    min((v["row"] for (sg, ag), v in winner_by_group.items() if sg == signal),
        key=selection_key)
    for signal in SIGNALS
]).reset_index(drop=True)

best_binary = best_per_signal.loc[best_per_signal.signal == "binary"].iloc[0].to_dict()
best_graded = best_per_signal.loc[best_per_signal.signal == "graded"].iloc[0].to_dict()
best_overall = min((best_binary, best_graded), key=selection_key)

print("Лучшие настройки по каждой из четырёх групп:")
display(best_four[["signal", "aggregation", "k_neighbors", *[f"eligible_{m}" for m in METRICS]]])
print("Baseline binary vs graded (оба оптимизированы по validation NDCG):")
display(best_per_signal[["signal", "aggregation", "k_neighbors", *[f"eligible_{m}" for m in METRICS]]])
print("Лучшая конфигурация среди всех 44:", {k: best_overall[k] for k in
      ("signal", "aggregation", "k_neighbors", "eligible_ndcg_at_10")})
require(best_overall["eligible_ndcg_at_10"] == results["eligible_ndcg_at_10"].max(),
        "Выбор модели не соответствует максимальному NDCG на validation")


plot_specs = [
    ("eligible_ndcg_at_10", "NDCG@10", "itemknn_val_ndcg_at_10.html"),
    ("eligible_recall_at_10", "Recall@10", "itemknn_val_recall_at_10.html"),
    ("eligible_map_at_10", "MAP@10", "itemknn_val_map_at_10.html"),
]
plot_figs = {}
for column, title, filename in plot_specs:
    fig = px.line(
        results.sort_values("k_neighbors"),
        x="k_neighbors", y=column,
        color="signal", line_dash="aggregation", markers=True,
        title=f"ItemKNN · Validation Eligible {title} vs число соседей k",
        labels={"k_neighbors": "Число соседей видео, k",
                column: f"Eligible {title}", "signal": "Train signal", "aggregation": "Scoring"},
        hover_data={"k_neighbors": True, "eligible_share": ":.3f", "top10_zero_score_fraction": ":.3f"},
    )
    fig.update_xaxes(type="log", tickmode="array", tickvals=list(K_GRID),
                     ticktext=[str(k) for k in K_GRID])
    fig.update_layout(legend_title_text="Signal / Aggregation", hovermode="x unified")
    plot_figs[filename] = fig
    fig.show()


bootstrap_rows = []
for (signal, aggregation), winner in winner_by_group.items():
    for metric in ("ndcg_at_10", "recall_at_10", "ap_at_10"):
        for group in ("eligible", "all_users"):
            ci = bootstrap_mean(winner["per_user"], metric=metric, group=group,
                                n_bootstrap=BOOTSTRAP_N, ci=0.95, seed=BOOTSTRAP_SEED)
            require(np.isfinite([ci["estimate"], ci["ci_low"], ci["ci_high"]]).all() and
                    ci["ci_low"] <= ci["ci_high"], "Некорректный CI")
            bootstrap_rows.append({
                "signal": signal, "aggregation": aggregation,
                "k_neighbors": winner["row"]["k_neighbors"],
                "metric": "map_at_10" if metric == "ap_at_10" else metric,
                "group": group, **ci,
            })
bootstrap_table = pd.DataFrame(bootstrap_rows)

def group_winner_for(row):
    return winner_by_group[(row["signal"], row["aggregation"])]

paired_rows = []
graded_users = group_winner_for(best_graded)["per_user"]
binary_users = group_winner_for(best_binary)["per_user"]
for metric in ("ndcg_at_10", "recall_at_10", "ap_at_10"):
    for group in ("eligible", "all_users"):
        d = bootstrap_paired_difference(graded_users, binary_users, metric=metric, group=group,
                                        n_bootstrap=BOOTSTRAP_N, ci=0.95, seed=BOOTSTRAP_SEED)
        paired_rows.append({"metric": "map_at_10" if metric == "ap_at_10" else metric,
                            "group": group, **d})
paired_table = pd.DataFrame(paired_rows)
print("Bootstrap CI для лучших четырёх конфигураций:")
display(bootstrap_table[["signal", "aggregation", "k_neighbors", "metric", "group",
                         "estimate", "ci_low", "ci_high"]])
print("Парная разность graded − binary (валидирована на одних и тех же eligible пользователях):")
display(paired_table[["metric", "group", "estimate", "ci_low", "ci_high"]])


OUT.mkdir(parents=True, exist_ok=True)
results.sort_values(["signal", "aggregation", "k_neighbors"]).to_csv(
    OUT / "all_44_validation_metrics.csv", index=False)
best_four.to_csv(OUT / "best_four_by_signal_and_aggregation.csv", index=False)
best_per_signal.to_csv(OUT / "best_binary_vs_graded.csv", index=False)
bootstrap_table.to_csv(OUT / "bootstrap_best_four.csv", index=False)
paired_table.to_csv(OUT / "paired_graded_minus_binary.csv", index=False)
label_table.to_csv(OUT / "relevance_counts.csv", index=False)
pd.DataFrame(fit_stats).to_csv(OUT / "fit_diagnostics.csv", index=False)
for name, fig in plot_figs.items():
    fig.write_html(str(OUT / name), include_plotlyjs=True, full_html=True)

for (signal, aggregation), winner in winner_by_group.items():
    stem = f"winner_{signal}_{aggregation}"
    per_user = winner["per_user"]
    rec = winner["recommendations"]
    require(len(per_user) == 3000 and len(rec) == 30000,
            "Ошибка сохранения: неверная форма данных у победителя")
    require(int(per_user.eligible.sum()) == 2082, "Другая eligible-маска у победителя")
    require(not pd.MultiIndex.from_frame(rec[["user_id", "item_id"]]).isin(
        pd.MultiIndex.from_frame(HISTORY[["user_id", "item_id"]])).any(),
        "В рекомендациях победителя есть ранее просмотренное видео")
    per_user.to_csv(OUT / f"{stem}_per_user.csv", index=False)
    rec.to_csv(OUT / f"{stem}_recommendations.csv", index=False)

selection = {
    "selection_metric": "validation eligible NDCG@10 with linear gains (0,1,2)",
    "tie_break": "higher NDCG, then smaller k_neighbors, then sum before normalized",
    "configurations_evaluated": 44,
    "neighbor_grid": list(K_GRID),
    "signals": list(SIGNALS),
    "aggregations": list(AGGREGATIONS),
    "best_binary": best_binary,
    "best_graded": best_graded,
    "best_overall": best_overall,
    "best_four": best_four.to_dict(orient="records"),
}
manifest = {
    "status": "PASS_VALIDATION_ONLY",
    "dataset": "deepvk/VK-LSVD", "subsample": "up0.001_ip0.001",
    "source_revision": REVISION["revision"],
    "user_ids_sha256": user_hash, "item_ids_sha256": item_hash,
    "input_file_sha256": file_hashes, "verified_module_sha256": module_hashes,
    "n_train": int(len(HISTORY)), "n_validation": int(len(HOLDOUT)),
    "n_users": int(len(USER_IDS)), "n_items": int(len(ITEM_IDS)),
    "train_weeks": "0–24", "validation_week": 25,
    "k_recommendations": 10, "n_eligible_val": 2082, "eligible_share": 2082 / 3000,
    "bootstrap_resamples": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
    "fit_diagnostics": fit_stats,
    "selection": selection,
    "validation_bootstrap_selection_bias_warning": True,
    "test_data_read": False,
    "test_metrics_calculated": False,
    "train_metrics_calculated": False,
}
(OUT / "selection_and_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
(OUT / "README_results.md").write_text(
    "# VK-LSVD / ItemKNN / Validation\n\n"
    "Выборка фиксирована по исходному EDA: 3000 пользователей / 2000 видео, "
    "train weeks 0–24, validation week 25. Все 44 варианта проверены на одном evaluator. "
    "Основная метрика: Eligible NDCG@10; дополнительные Recall@10 и MAP@10. "
    "Test не открывался.\n\n"
    "Главные файлы: `all_44_validation_metrics.csv`, "
    "`best_binary_vs_graded.csv`, `selection_and_manifest.json`. "
    "HTML — интерактивные графики. "
    "`winner_*_per_user.csv` и `winner_*_recommendations.csv` — четыре выбранных "
    "варианта (по сигналу и способу скоринга). "
    "Все широкие ID при чтении CSV должны оставаться uint64. "
    "Bootstrap CI после выбора на validation — диагностические, а не итоговые test CI.\n",
    encoding="utf-8",
)
require(len(list(OUT.glob("winner_*_per_user.csv"))) == 4, "Не все per-user таблицы сохранены")
require(len(list(OUT.glob("winner_*_recommendations.csv"))) == 4, "Не все рекомендации сохранены")
print("PASS: все 44 результата и артефакты сохранены:", OUT)
print("Финальный тест не открывался. Переобучения на train+val пока нет.")
print('Результаты сохранены.')