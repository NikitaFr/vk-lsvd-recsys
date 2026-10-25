"""relevance experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, Markdown

ROOT = Path.cwd()
WORK = ROOT / "data" / "work"
OUT = ROOT / "results" / "relevance_diagnostics"
TAB = OUT / "tables"
FIG = OUT / "figures"
for folder in (TAB, FIG):
    folder.mkdir(parents=True, exist_ok=True)

required = [WORK / "interactions.parquet", WORK / "items_metadata.parquet"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Не найдены рабочие Parquet-файлы. Запусти Jupyter из корня первого проекта. Нет: "
        + ", ".join(missing)
    )

TABLE_INDEX, FIG_INDEX = [], []
summary = {"purpose": "Exploratory relevance diagnostics; no labels fixed for evaluation"}

def export(name, table, purpose):
    out = TAB / f"{name}.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    TABLE_INDEX.append({"file": out.name, "purpose": purpose, "rows": len(table)})
    return table

def show(name, fig, purpose):
    fig.update_layout(template="plotly_white", height=455,
                      margin=dict(l=60, r=40, t=70, b=70), hovermode="closest")
    fig.write_html(str(FIG / f"{name}.html"), include_plotlyjs="directory", full_html=True)
    FIG_INDEX.append({"file": f"{name}.html", "purpose": purpose})
    display(fig)

def json_native(o):
    if isinstance(o, dict):
        return {str(k): json_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_native(x) for x in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o

REACTS = ["like", "bookmark", "dislike"]
DEPTH_NAMES = ["<10%", "10–20%", "20–50%", "50–80%", "80–100%", "≥100%"]
DEPTH_EDGES = [0, .10, .20, .50, .80, 1.0, np.inf]
DURATION_NAMES = ["1–15с", "16–30с", "31–60с", "61–120с", "121–180с", ">180с"]
DURATION_EDGES = [0, 15, 30, 60, 120, 180, np.inf]


events = pd.read_parquet(WORK / "interactions.parquet")
items = pd.read_parquet(WORK / "items_metadata.parquet")
required_event_cols = {"user_id", "item_id", "timespent", *REACTS}
missing_cols = required_event_cols - set(events.columns)
if missing_cols:
    raise ValueError(f"В events нет нужных колонок: {sorted(missing_cols)}")
if not {"item_id", "duration"}.issubset(items.columns):
    raise ValueError("В items_metadata ожидаются 'item_id' и 'duration'")
if items.item_id.duplicated().any():
    raise ValueError("Дубли item_id в метаданных: присоединение длительности неоднозначно")
if events.duplicated(["user_id", "item_id"]).any():
    raise ValueError("Повторные пары user_id-item_id: требуется проверить структуру выборки")

d = events[["user_id", "item_id", "timespent", *REACTS]].copy()
for col in ["timespent", *REACTS]:
    d[col] = pd.to_numeric(d[col], errors="coerce")
for col in REACTS:
    if d[col].isna().any() or not d[col].isin([0, 1]).all():
        raise ValueError(f"Неожиданные значения реакции {col}: требуется проверка")
if d.timespent.isna().any() or (d.timespent < 0).any():
    raise ValueError("Время просмотра отсутствует или отрицательно")
d[REACTS] = d[REACTS].astype("bool")
d["duration"] = d.item_id.map(items.set_index("item_id").duration)
if d.duration.isna().any() or (d.duration <= 0).any():
    raise ValueError("Не удалось получить положительную duration для всех видео")
d["depth"] = d.timespent / d.duration
d["depth_group"] = pd.cut(d.depth, DEPTH_EDGES, labels=DEPTH_NAMES,
                          right=False, include_lowest=True)
d["duration_group"] = pd.cut(d.duration, DURATION_EDGES, labels=DURATION_NAMES,
                             right=True, include_lowest=True)

n_events = len(d)
n_users = d.user_id.nunique()
n_items = d.item_id.nunique()
assert d.depth.notna().all()
assert n_events > 0
integrity = {"events": n_events, "users": n_users, "items": n_items,
             "like_count": int(d["like"].sum()),
             "bookmark_count": int(d["bookmark"].sum()),
             "dislike_count": int(d["dislike"].sum()),
             "watch_depth_ge_80_count": int((d.depth >= .8).sum()),
             "observed_pairs_unique": bool(not d.duplicated(["user_id", "item_id"]).any()),
             "metadata_coverage": float(d.duration.notna().mean()),
             "assignment_size_ok": bool(100_000 <= n_events <= 2_000_000 and n_users >= 1000 and n_items >= 1000)}
summary["integrity"] = integrity
export("00_integrity", pd.DataFrame([integrity]), "Размер и согласованность рабочей выборки")
display(pd.Series(integrity, name="value").to_frame())


# Непересекающиеся интервалы глубины: легче оценивать rate и число событий.
reaction_by_depth = (
    d.groupby("depth_group", observed=False)
     .agg(n_events=("like", "size"), likes=("like", "sum"),
          bookmarks=("bookmark", "sum"), dislikes=("dislike", "sum"),
          median_watch_seconds=("timespent", "median"))
     .reset_index()
)
for r in REACTS:
    reaction_by_depth[f"{r}_rate"] = reaction_by_depth[{"like":"likes", "bookmark":"bookmarks", "dislike":"dislikes"}[r]] / reaction_by_depth.n_events.replace(0,np.nan)
reaction_by_depth = export("01_reactions_by_depth", reaction_by_depth,
                           "Число показов и реакций по непересекающимся интервалам доли просмотра")
display(reaction_by_depth)

# Пересекающиеся кумулятивные пороги: все короткие просмотры строго ниже границы.
shallow_rows = []
for threshold in [.10, .20, .50]:
    mask = d.depth < threshold
    r = {"depth_strictly_below": threshold, "n_events": int(mask.sum()),
         "event_share_all": float(mask.mean())}
    for react in ("like", "bookmark"):
        n = int((mask & d[react]).sum())
        r[f"{react}_count"] = n
        r[f"share_of_all_{react}s"] = n / int(d[react].sum()) if d[react].sum() else np.nan
        r[f"{react}_rate_within_shallow_events"] = n / int(mask.sum()) if mask.sum() else np.nan
    r["likes_with_watch_lt_3s"] = int((mask & d.like & (d.timespent < 3)).sum())
    r["bookmarks_with_watch_lt_3s"] = int((mask & d.bookmark & (d.timespent < 3)).sum())
    shallow_rows.append(r)
shallow = export("01_shallow_positive_signals", pd.DataFrame(shallow_rows),
                 "Кумулятивная частота лайков и сохранений среди очень коротких просмотров")
display(shallow)

fig = go.Figure()
for name, cnt, color in [("Лайки", "likes", "#327CB1"),
                         ("Избранное", "bookmarks", "#D68A32")]:
    fig.add_trace(go.Bar(x=reaction_by_depth.depth_group.astype(str),
                         y=100*reaction_by_depth[cnt]/reaction_by_depth.n_events.replace(0,np.nan),
                         name=name, marker_color=color,
                         customdata=np.column_stack((reaction_by_depth[cnt], reaction_by_depth.n_events)),
                         hovertemplate="Доля просмотра: %{x}<br>Rate: %{y:.3f}%<br>Реакций: %{customdata[0]}<br>Показов: %{customdata[1]}<extra></extra>"))
fig.update_layout(title="Явные положительные реакции по глубине просмотра",
                  xaxis_title="Доля просмотренного видео", yaxis_title="Реакции / показы, %", barmode="group")
show("01_positive_reactions_by_depth", fig,
     "Частота лайков и избранного по интервалам просмотра с размерами групп")
summary["shallow_reactions"] = shallow_rows


conflict_rows=[]
def conflict(name, mask):
    conflict_rows.append({"case":name, "n_events":int(mask.sum()),
                          "share_all_events":float(mask.mean()),
                          "share_among_dislikes":float(mask.sum()/d.dislike.sum()) if d.dislike.sum() else None})
conflict("Любой дизлайк", d.dislike)
conflict("Дизлайк и просмотр ≥80%", d.dislike & (d.depth >= .8))
conflict("Дизлайк и просмотр ≥90%", d.dislike & (d.depth >= .9))
conflict("Дизлайк и лайк", d.dislike & d.like)
conflict("Дизлайк и избранное", d.dislike & d.bookmark)
conflict("Дизлайк и (лайк или избранное)", d.dislike & (d.like | d.bookmark))
conflict("Дизлайк и просмотр <10%", d.dislike & (d.depth < .1))
conflict("Лайк и избранное", d.like & d.bookmark)
conflicts = export("02_reaction_conflicts", pd.DataFrame(conflict_rows),
                   "Наблюдаемые конфликтные и совместные реакции, с учётом знаменателей")
display(conflicts)
summary["conflicts"] = conflict_rows


deep = d[d.depth >= .8].copy()
sec_edges = [0, 3, 5, 10, 20, 30, 60, 120, 256, np.inf]
sec_names = ["0–<3", "3–<5", "5–<10", "10–<20", "20–<30", "30–<60", "60–<120", "120–255", ">255"]
deep["seconds_group"] = pd.cut(deep.timespent, sec_edges, labels=sec_names,
                                right=False, include_lowest=True)
seconds_summary = (deep.groupby("seconds_group", observed=False)
                   .agg(n_events=("timespent", "size"),
                        likes=("like", "sum"), bookmarks=("bookmark", "sum"))
                   .reset_index())
seconds_summary["share_among_deep"] = seconds_summary.n_events / len(deep) if len(deep) else np.nan
seconds_summary = export("03_watch_seconds_among_deep", seconds_summary,
                         "Длительность фактического просмотра для взаимодействий с глубиной ≥80%")
display(seconds_summary)

dur_deep=[]
for label in DURATION_NAMES:
    chunk = d[d.duration_group.astype(str) == label]
    m = chunk.depth >= .8
    v = chunk.loc[m, "timespent"]
    dur_deep.append({"duration_group":label, "n_events":int(len(chunk)),
                     "n_depth_ge80":int(m.sum()),
                     "share_depth_ge80":float(m.mean()) if len(chunk) else None,
                     "deep_watch_seconds_median":float(v.median()) if len(v) else None,
                     "deep_watch_seconds_p10":float(v.quantile(.1)) if len(v) else None,
                     "deep_watch_seconds_p90":float(v.quantile(.9)) if len(v) else None,
                     "deep_watch_lt3s_share":float((v < 3).mean()) if len(v) else None,
                     "deep_watch_lt5s_share":float((v < 5).mean()) if len(v) else None,
                     "deep_watch_lt10s_share":float((v < 10).mean()) if len(v) else None})
by_duration = export("03_deep_watch_by_duration", pd.DataFrame(dur_deep),
                     "Время просмотра среди глубоких просмотров внутри групп длительности видео")
display(by_duration)

fig = go.Figure(go.Bar(x=seconds_summary.seconds_group.astype(str),
                       y=100*seconds_summary.share_among_deep,
                       customdata=seconds_summary.n_events,
                       marker_color="#327CB1",
                       hovertemplate="Секунды: %{x}<br>Доля глубоких просмотров: %{y:.2f}%<br>N=%{customdata}<extra></extra>"))
fig.update_layout(title="Сколько секунд длится просмотр глубиной ≥80%?",
                  xaxis_title="Просмотрено секунд", yaxis_title="Доля глубоких просмотров, %")
show("03_absolute_seconds_among_deep", fig,
     "Распределение абсолютного времени у просмотров с глубиной ≥80%")
summary["deep_watch"] = {"n_depth_ge80":int(len(deep)),
                         "deep_watch_lt3s":int((deep.timespent < 3).sum()),
                         "deep_watch_lt5s":int((deep.timespent < 5).sum()),
                         "deep_watch_lt10s":int((deep.timespent < 10).sum()),
                         "deep_seconds_median":float(deep.timespent.median()) if len(deep) else None}


scenario_spec = {
    "A_lenient": {"moderate_depth": .80, "moderate_sec": 0, "explicit_sec": 0, "dislike_veto": False},
    "B_balanced": {"moderate_depth": .80, "moderate_sec": 5, "explicit_sec": 3, "dislike_veto": False},
    "C_strict": {"moderate_depth": .90, "moderate_sec": 10, "explicit_sec": 5, "dislike_veto": False},
    "D_balanced_veto": {"moderate_depth": .80, "moderate_sec": 5, "explicit_sec": 3, "dislike_veto": True},
}
user_ids = pd.Index(d.user_id.unique(), name="user_id")
item_ids = pd.Index(d.item_id.unique(), name="item_id")
user_result = pd.DataFrame({"user_id":user_ids})
item_result = pd.DataFrame({"item_id":item_ids})
rows=[]
by_duration_rows=[]

for name, spec in scenario_spec.items():
    strong = (d.like | d.bookmark) & (d.timespent >= spec["explicit_sec"])
    moderate = (d.depth >= spec["moderate_depth"]) & (d.timespent >= spec["moderate_sec"])
    # Приоритет strong > moderate > zero
    grades = np.where(strong, 2, np.where(moderate, 1, 0)).astype("uint8")
    if spec["dislike_veto"]:
        grades = np.where(d.dislike.to_numpy(), 0, grades).astype("uint8")
    positive = grades >= 1
    strong_only = grades == 2
    assert np.isin(grades, [0, 1, 2]).all()
    assert np.sum(positive) == np.sum(grades == 1) + np.sum(strong_only)

    # Пара user/item счётчиков, включая ID с нулём релевантных событий.
    counts_users = (pd.DataFrame({"user_id":d.user_id.to_numpy(), "positive":positive,
                                  "strong":strong_only})
                   .groupby("user_id", observed=True)[["positive", "strong"]].sum()
                   .reindex(user_ids, fill_value=0))
    counts_items = (pd.DataFrame({"item_id":d.item_id.to_numpy(), "positive":positive,
                                  "strong":strong_only})
                   .groupby("item_id", observed=True)[["positive", "strong"]].sum()
                   .reindex(item_ids, fill_value=0))
    assert int(counts_users.positive.sum()) == int(positive.sum())
    assert int(counts_items.positive.sum()) == int(positive.sum())
    user_result[name+"_positive"] = counts_users.positive.to_numpy()
    user_result[name+"_strong"] = counts_users.strong.to_numpy()
    item_result[name+"_positive"] = counts_items.positive.to_numpy()
    item_result[name+"_strong"] = counts_items.strong.to_numpy()
    rows.append({"scenario":name, "level0":int((grades == 0).sum()),
                 "level1":int((grades == 1).sum()), "level2":int((grades == 2).sum()),
                 "n_positive":int(positive.sum()), "share_positive":float(positive.mean()),
                 "share_level2":float(strong_only.mean()),
                 "users_positive_ge1":int((counts_users.positive >= 1).sum()),
                 "users_positive_ge5":int((counts_users.positive >= 5).sum()),
                 "users_positive_ge10":int((counts_users.positive >= 10).sum()),
                 "users_strong_ge1":int((counts_users.strong >= 1).sum()),
                 "items_positive_ge1":int((counts_items.positive >= 1).sum()),
                 "items_positive_ge5":int((counts_items.positive >= 5).sum()),
                 "items_positive_ge10":int((counts_items.positive >= 10).sum()),
                 "items_strong_ge1":int((counts_items.strong >= 1).sum()),
                 "median_positive_per_user":float(counts_users.positive.median()),
                 "median_positive_per_item":float(counts_items.positive.median()),
                 "users_zero_positive":int((counts_users.positive == 0).sum()),
                 "items_zero_positive":int((counts_items.positive == 0).sum())})
    sub = pd.DataFrame({"group":d.duration_group, "grade":grades})
    agg = (sub.groupby("group", observed=False).grade
           .agg(n_events="size", positive=lambda v: int((v >= 1).sum()),
                strong=lambda v: int((v == 2).sum())).reset_index())
    for _,r in agg.iterrows():
        by_duration_rows.append({"scenario":name, "duration_group":str(r["group"]),
                                 "n_events":int(r["n_events"]),
                                 "n_positive":int(r["positive"]),
                                 "positive_rate":float(r["positive"]/r["n_events"]) if r["n_events"] else None,
                                 "n_strong":int(r["strong"])})

scenario_table = export("04_candidate_scenarios", pd.DataFrame(rows),
                         "Распределения уровней 0/1/2 и покрытие пользователей/видео для 4 гипотетических правил")
export("04_candidate_user_coverage", user_result,
       "На каждого пользователя число положительных и сильных событий для каждого сценария")
export("04_candidate_item_coverage", item_result,
       "На каждое видео число положительных и сильных событий для каждого сценария")
by_duration_table = export("04_candidate_by_duration", pd.DataFrame(by_duration_rows),
                           "Распределение положительных событий внутри групп длительности для каждого сценария")
display(scenario_table)

fig = go.Figure()
for col,color,label in [("level1","#327CB1","Уровень 1"),
                        ("level2","#D68A32","Уровень 2")]:
    fig.add_trace(go.Bar(x=scenario_table.scenario, y=scenario_table[col],
                         name=label, marker_color=color))
fig.update_layout(title="Сколько положительных событий остаётся при разных правилах?",
                  xaxis_title="Пробный критерий", yaxis_title="Число наблюдаемых событий", barmode="stack")
show("04_scenarios_positive_counts", fig,
     "Число наблюдаемых пар уровня 1 и 2 для каждого пробного критерия")

fig = go.Figure()
for col,label,color in [("users_positive_ge5","Пользователей с ≥5", "#327CB1"),
                         ("items_positive_ge5","Видео с ≥5", "#D68A32")]:
    fig.add_trace(go.Bar(x=scenario_table.scenario, y=scenario_table[col],
                         name=label, marker_color=color))
fig.update_layout(title="Покрытие пользователей и видео при разных правилах",
                  xaxis_title="Пробный критерий", yaxis_title="Количество ID с ≥5 положительными событиями",
                  barmode="group")
show("04_scenarios_coverage", fig,
     "Покрытие пользователей и объектов, при котором у ID не менее пяти положительных событий")
summary["scenarios"] = rows
summary["scenario_definitions"] = scenario_spec


assert sum(reaction_by_depth.n_events) == n_events, "Не все события учтены в depth bins"
assert int(reaction_by_depth.likes.sum()) == int(d.like.sum())
assert int(reaction_by_depth.bookmarks.sum()) == int(d.bookmark.sum())
assert int(seconds_summary.n_events.sum()) == len(deep)
assert (scenario_table[["level0","level1","level2"]].sum(axis=1) == n_events).all()
assert all((scenario_table.users_positive_ge1 <= n_users) &
           (scenario_table.items_positive_ge1 <= n_items))
summary["controls"] = {"depth_bins_sum_to_events":True,
                       "deep_seconds_bins_sum_to_deep_events":True,
                       "scenario_levels_sum_to_events":True,
                       "source_parquets_modified":False}
summary["tables"] = TABLE_INDEX
summary["figures"] = FIG_INDEX
(OUT / "summary.json").write_text(
    json.dumps(json_native(summary), ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "README_results.md").write_text(
    "# Диагностика релевантности VK-LSVD\n\n"
    "Пять диагностических вопросов: ранние лайки, ранние сохранения, противоречия с дизлайками, "
    "абсолютные секунды глубоких просмотров, последствия четырёх пробных шкал 0/1/2.\n\n"
    "`tables/` — CSV с агрегатами и числом наблюдений; `figures/` — Plotly HTML; "
    "`summary.json` — машинная сводка.\n\n"
    "Сценарии — исследовательские и не являются автоматически выбранными метками test. "
    "Ненаблюдаемые пары не маркируются негативными.\n",
    encoding="utf-8")
print(f"✅ Готово. Результаты: {OUT.relative_to(ROOT)}")
print(f"CSV: {len(TABLE_INDEX)}; HTML-графики: {len(FIG_INDEX)}")
print('Результаты сохранены.')