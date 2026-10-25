"""eda checks experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

ROOT = Path.cwd()
WORK = ROOT / "data" / "work"
OUT = ROOT / "results" / "eda_final_checks"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
for directory in (TABLES, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

SCORE_LEVELS = (50, 80, 90, 100)
PRIMARY_LEVEL = 90  # Для наиболее подробных графиков, не для модельного таргета
FILES = {
    "events": WORK / "interactions.parquet",
    "items": WORK / "items_metadata.parquet",
    "users": WORK / "users_metadata.parquet",
}
missing_files = [str(path) for path in FILES.values() if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        "Рабочие данные первого EDA не найдены. Запускай Jupyter из корня проекта. "
        "Отсутствуют: " + ", ".join(missing_files)
    )

saved_tables = []
saved_figures = []

def export_table(stem, frame, meaning):
    path = TABLES / f"{stem}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    saved_tables.append({"file": path.name, "rows": len(frame), "meaning": meaning})
    return frame

def display_figure(stem, figure, meaning, height=485):
    figure.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=65, r=30, t=80, b=75),
        hovermode="closest",
        font=dict(size=13),
    )
    figure.write_html(FIGURES / f"{stem}.html", include_plotlyjs="directory", full_html=True)
    saved_figures.append({"file": f"{stem}.html", "meaning": meaning})
    display(figure)

def as_jsonable(value):
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or pd.isna(value):
        return None
    return value

print("Корень проекта:", ROOT)
print("Результаты:", OUT)


# Из Parquet читаем только нужные для проверки колонки.
events = pd.read_parquet(FILES["events"], columns=["user_id", "item_id", "week", "timespent"])
items = pd.read_parquet(FILES["items"], columns=["item_id", "duration"])
users = pd.read_parquet(FILES["users"], columns=["user_id"])

if not events["week"].notna().all():
    raise ValueError("Присутствуют события без номера недели")
weeks_raw = pd.to_numeric(events["week"], errors="coerce")
if weeks_raw.isna().any() or not np.array_equal(weeks_raw.to_numpy(), weeks_raw.astype("int64").to_numpy()):
    raise ValueError("Недели должны быть целыми числами")
if not weeks_raw.between(0, 24).all():
    raise ValueError("Для EDA разрешены только исторические недели 0..24; другие недели не анализируем")
events["week"] = weeks_raw.astype("int16")

if items.item_id.duplicated().any() or users.user_id.duplicated().any():
    raise ValueError("Дублируются идентификаторы в метаданных")
if events.duplicated(["user_id", "item_id"]).any():
    raise ValueError("Повторяются пары пользователь–видео; остановка до выяснения причины")
if events.user_id.isna().any() or events.item_id.isna().any():
    raise ValueError("Есть взаимодействия без ID")
if not events.user_id.isin(users.user_id).all():
    raise ValueError("Часть пользователей не имеет метаданных")
if not events.item_id.isin(items.item_id).all():
    raise ValueError("Часть видео не имеет метаданных")

duration = pd.to_numeric(items.duration, errors="coerce")
if duration.isna().any() or (duration <= 0).any():
    raise ValueError("Есть некорректные длительности видео. Нужно отдельно разобрать их")
watch = pd.to_numeric(events.timespent, errors="coerce")
if watch.isna().any() or (watch < 0).any():
    raise ValueError("Есть некорректные значения времени просмотра")

# map без размножения строк; ничего не изменяем в исходных файлах.
d = events[["user_id", "item_id", "week"]].copy()
d["ratio"] = watch.to_numpy() / d.item_id.map(items.set_index("item_id").duration).to_numpy()
if not np.isfinite(d.ratio.to_numpy()).all():
    raise ValueError("Не удалось вычислить глубину просмотра для всех событий")
for level in SCORE_LEVELS:
    d[f"watch_ge_{level}"] = d.ratio >= level / 100

user_ids = pd.Index(users.user_id.unique(), name="user_id")
item_ids = pd.Index(items.item_id.unique(), name="item_id")
weeks = pd.Index(range(25), name="week")

checks = {
    "interactions": len(d),
    "users_in_metadata": len(user_ids),
    "items_in_metadata": len(item_ids),
    "distinct_event_users": int(d.user_id.nunique()),
    "distinct_event_items": int(d.item_id.nunique()),
    "historical_weeks": [int(x) for x in sorted(d.week.unique())],
    "watch_ratio_gt_one": int((d.ratio > 1).sum()),
}
# Эти числа применимы именно к ранее выбранной рабочей подвыборке.
print(pd.Series(checks).to_string())
if not (100_000 <= len(d) <= 2_000_000 and len(user_ids) >= 1_000 and len(item_ids) >= 1_000):
    print("ВНИМАНИЕ: рабочие данные не удовлетворяют численным ограничениям задания")


all_item_counts = d.groupby("item_id", sort=False).agg(n_events=("ratio", "size"))
for level in SCORE_LEVELS:
    all_item_counts[f"watch_ge_{level}"] = d.groupby("item_id")[f"watch_ge_{level}"].sum()
# Включаем каталог полностью, даже если какие-то видео не встречались в событиях.
item_table = all_item_counts.reindex(item_ids, fill_value=0).reset_index()
for level in SCORE_LEVELS:
    item_table[f"rate_ge_{level}"] = np.where(
        item_table.n_events.gt(0),
        item_table[f"watch_ge_{level}"] / item_table.n_events.replace(0, np.nan),
        np.nan,
    )
item_table = export_table(
    "01_item_signal_counts", item_table,
    "Число наблюдений и глубоких просмотров по каждому видео, включая нули"
)

item_stats = []
for level in SCORE_LEVELS:
    c = item_table[f"watch_ge_{level}"]
    item_stats.append({
        "threshold_pct": level,
        "n_items": int(len(c)),
        "n_items_zero": int((c == 0).sum()),
        "n_items_ge1": int((c >= 1).sum()),
        "n_items_ge5": int((c >= 5).sum()),
        "n_items_ge10": int((c >= 10).sum()),
        "share_items_ge10": float((c >= 10).mean()),
        "item_count_p10": float(c.quantile(.1)),
        "item_count_median": float(c.median()),
        "item_count_p90": float(c.quantile(.9)),
        "total_signals": int(c.sum()),
        "share_of_events": float(c.sum() / len(d)),
    })
item_coverage = export_table(
    "01_item_coverage_summary", pd.DataFrame(item_stats),
    "Покрытие каталога по разным порогам глубокого просмотра"
)
display(item_coverage)

# Для оценки концентрации сравниваем те же топ-10% объектов по числу показов
# с топ-10% по числу глубоких просмотров.
top_n = max(1, int(np.ceil(len(item_table) * .1)))
by_pop = item_table.sort_values(["n_events", "item_id"], ascending=[False, True]).head(top_n)
by_deep = item_table.sort_values(["watch_ge_90", "item_id"], ascending=[False, True]).head(top_n)
all_90 = int(item_table.watch_ge_90.sum())
concentration = {
    "top10pct_item_count": top_n,
    "share_all_events_on_top_popular_items": float(by_pop.n_events.sum() / len(d)),
    "share_deep90_on_top_popular_items": float(by_pop.watch_ge_90.sum() / all_90) if all_90 else None,
    "share_deep90_on_top_deep_items": float(by_deep.watch_ge_90.sum() / all_90) if all_90 else None,
    "min_item_events": int(item_table.n_events.min()),
    "median_item_events": float(item_table.n_events.median()),
}
print("Покрытие и концентрация:")
display(pd.Series(concentration, name="value").to_frame())

# Важные численные инварианты: разбиение по видео не теряет событий.
assert item_table.n_events.sum() == len(d)
for level in SCORE_LEVELS:
    assert item_table[f"watch_ge_{level}"].sum() == int(d[f"watch_ge_{level}"].sum())


# График 1: сколько глубоких просмотров приходится на одно видео.
fig = go.Figure()
for level, color in [(80, "#337DA8"), (90, "#D98A36")]:
    values = np.sort(item_table[f"watch_ge_{level}"].to_numpy())
    y = np.arange(1, len(values) + 1) / len(values)
    fig.add_trace(go.Scatter(
        x=values, y=y, mode="lines", line=dict(shape="hv", width=2, color=color),
        name=f"Глубина ≥{level}%", hovertemplate="Сигналов на видео: %{x}<br>Доля видео с ≤ этого числа: %{y:.1%}<extra></extra>"
    ))
fig.update_layout(
    title="Распределение числа глубоких просмотров по всем видео (ECDF)",
    xaxis_title="Число глубоких просмотров на одно видео",
    yaxis_title="Доля видео с не более X просмотрами", yaxis_tickformat=".0%",
)
display_figure("01_item_deep_count_ecdf", fig, "ECDF глубоких просмотров среди видео, нулевые значения учитываются")

# График 2: не смешиваем популярность объекта с долей глубоких просмотров.
fig = go.Figure()
for mask, name, color, alpha in [
    (item_table.n_events < 100, "<100 показов (оценка шумнее)", "#D98A36", .5),
    (item_table.n_events >= 100, "≥100 показов", "#337DA8", .7),
]:
    x = item_table.loc[mask]
    if len(x):
        fig.add_trace(go.Scattergl(
            x=x.n_events, y=x.rate_ge_90 * 100, mode="markers", name=name,
            marker=dict(size=7, color=color, opacity=alpha),
            customdata=np.column_stack((x.item_id.astype(str), x.watch_ge_90)),
            hovertemplate="ID видео: %{customdata[0]}<br>Показов: %{x}<br>Глубоких ≥90%: %{customdata[1]}<br>Доля: %{y:.1f}%<extra></extra>",
        ))
fig.update_layout(
    title="Популярность видео и доля просмотров ≥90%",
    xaxis_title="Число показов видео", yaxis_title="Глубокие просмотры / все показы, %",
)
display_figure("01_item_popularity_vs_deep_rate", fig,
               "Зависимость доли глубоких просмотров от популярности и размера выборки по каждому видео")


# Формируем полный набор user x week (для 3 000 пользователей ~75 тыс. строк).
per_week = (d.groupby(["user_id", "week"], observed=True)
            .agg(n_events=("ratio", "size"),
                 watch_ge_80=("watch_ge_80", "sum"),
                 watch_ge_90=("watch_ge_90", "sum"))
            .reindex(pd.MultiIndex.from_product([user_ids, weeks], names=["user_id", "week"]),
                     fill_value=0).reset_index())
per_week[["n_events", "watch_ge_80", "watch_ge_90"]] = (
    per_week[["n_events", "watch_ge_80", "watch_ge_90"]].astype("int32")
)
per_week["active"] = per_week.n_events > 0
per_week["any_ge_80"] = per_week.watch_ge_80 >= 1
per_week["any_ge_90"] = per_week.watch_ge_90 >= 1
per_week["ge5_ge_90"] = per_week.watch_ge_90 >= 5
export_table("02_user_week_counts", per_week,
             "Полная таблица пользователь-неделя, включая нулевую активность; позволяет проверить все агрегаты")

all_n_users = len(user_ids)
week_summary = (per_week.groupby("week")
                .agg(n_active_users=("active", "sum"),
                     n_users_ge1_80=("any_ge_80", "sum"),
                     n_users_ge1_90=("any_ge_90", "sum"),
                     n_users_ge5_90=("ge5_ge_90", "sum"),
                     n_events=("n_events", "sum"),
                     n_watch_ge_80=("watch_ge_80", "sum"),
                     n_watch_ge_90=("watch_ge_90", "sum"))
                .reset_index())
week_summary["n_users_total"] = all_n_users
for label in ("n_active_users", "n_users_ge1_80", "n_users_ge1_90", "n_users_ge5_90"):
    week_summary[f"{label}_share_all"] = week_summary[label] / all_n_users
for label in ("n_users_ge1_80", "n_users_ge1_90", "n_users_ge5_90"):
    week_summary[f"{label}_share_active"] = week_summary[label] / week_summary.n_active_users.replace(0, np.nan)
week_summary["event_rate_ge90"] = week_summary.n_watch_ge_90 / week_summary.n_events.replace(0, np.nan)
week_summary = export_table("02_weekly_coverage", week_summary,
                            "Недельное покрытие глубокими просмотрами: отдельно среди активных и всех пользователей")
display(week_summary[["week", "n_events", "n_active_users", "n_users_ge1_90",
                      "n_users_ge5_90", "n_users_ge1_90_share_active",
                      "n_users_ge1_90_share_all"]].head(10))

user_summary = (per_week.groupby("user_id")
                .agg(n_events=("n_events", "sum"),
                     n_active_weeks=("active", "sum"),
                     n_weeks_ge1_80=("any_ge_80", "sum"),
                     n_weeks_ge1_90=("any_ge_90", "sum"),
                     n_weeks_ge5_90=("ge5_ge_90", "sum"),
                     n_watch_ge80=("watch_ge_80", "sum"),
                     n_watch_ge90=("watch_ge_90", "sum"))
                .reset_index())
user_summary["fraction_active_weeks_with_ge90"] = (
    user_summary.n_weeks_ge1_90 / user_summary.n_active_weeks.replace(0, np.nan)
)
user_summary = export_table("02_user_temporal_coverage", user_summary,
                            "Для каждого пользователя: число активных недель и недель с глубокими просмотрами")

# Числовые инварианты предотвращают ошибки из-за неверной индексации или недостающих нулей.
assert len(per_week) == len(user_ids) * 25
assert int(per_week.n_events.sum()) == len(d)
assert int(per_week.watch_ge_90.sum()) == int(d.watch_ge_90.sum())
assert int(week_summary.n_watch_ge_90.sum()) == int(d.watch_ge_90.sum())
assert int(user_summary.n_watch_ge90.sum()) == int(d.watch_ge_90.sum())

user_temporal_stats = {
    "n_users": len(user_ids),
    "n_weeks": 25,
    "n_users_with_no_active_weeks": int((user_summary.n_active_weeks == 0).sum()),
    "n_users_with_zero_weeks_ge90": int((user_summary.n_weeks_ge1_90 == 0).sum()),
    "median_active_weeks": float(user_summary.n_active_weeks.median()),
    "median_weeks_with_ge90": float(user_summary.n_weeks_ge1_90.median()),
    "p10_weeks_with_ge90": float(user_summary.n_weeks_ge1_90.quantile(.1)),
    "p90_weeks_with_ge90": float(user_summary.n_weeks_ge1_90.quantile(.9)),
    "median_fraction_active_weeks_with_ge90": float(user_summary.fraction_active_weeks_with_ge90.median()),
    "week_min_share_active_ge90": float(week_summary.n_users_ge1_90_share_active.min()),
    "week_max_share_active_ge90": float(week_summary.n_users_ge1_90_share_active.max()),
}
print("Результаты проверки временного покрытия:")
display(pd.Series(user_temporal_stats).to_frame("value"))


# График 3. Сколько недель на пользователя приходится с глубокими просмотрами?
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=user_summary.n_weeks_ge1_90, xbins=dict(start=-0.5, end=25.5, size=1),
    marker_color="#337DA8", name="Недели с просмотром ≥90%",
    hovertemplate="Недели: %{x}<br>Пользователей: %{y}<extra></extra>"
))
fig.update_layout(
    title="В скольких неделях у пользователя есть просмотр ≥90%?",
    xaxis_title="Число недель с ≥1 глубоким просмотром (из 25)",
    yaxis_title="Число пользователей", bargap=.12, showlegend=False,
)
display_figure("02_user_weeks_with_deep_watch", fig,
               "Распределение недель с глубокими просмотрами по всем пользователям")

# График 4. Разделяем активность и наличие глубоких просмотров (оба — % всех пользователей).
fig = go.Figure()
for col, name, color in [
    ("n_active_users_share_all", "Были любые показы", "#969EA8"),
    ("n_users_ge1_90_share_all", "Есть ≥1 просмотр ≥90%", "#337DA8"),
    ("n_users_ge5_90_share_all", "Есть ≥5 просмотров ≥90%", "#D98A36"),
]:
    fig.add_trace(go.Scatter(
        x=week_summary.week, y=100 * week_summary[col], mode="lines+markers",
        line=dict(width=2.5, color=color), marker=dict(size=5), name=name,
        hovertemplate="Неделя: %{x}<br>Доля всех пользователей: %{y:.1f}%<extra></extra>"
    ))
fig.update_layout(
    title="Недельное покрытие: активность и наличие глубоких просмотров",
    xaxis_title="Неделя (0–24)", yaxis_title="Доля всех пользователей, %",
    xaxis=dict(dtick=1), yaxis=dict(range=[0, 105]),
)
display_figure("02_weekly_user_coverage", fig,
               "Доли пользователей с любой активностью и с глубокими просмотрами по неделям")


summary = {
    "dataset": "VK-LSVD, ранее выбранная подвыборка, исторические недели 0–24",
    "checks": checks,
    "method": {
        "levels_percent": list(SCORE_LEVELS),
        "primary_visualization_level": PRIMARY_LEVEL,
        "threshold_comparison": "watch_ratio >= threshold; ratio > 1 не ограничивается",
        "week_definition": "исторические номера недели 0–24",
        "inactive_user_week": "0 просмотров, учитывается отдельно от отсутствия глубоких просмотров",
        "no_new_split_or_target": True,
    },
    "item_coverage": item_coverage.to_dict(orient="records"),
    "item_concentration": concentration,
    "user_temporal_coverage": user_temporal_stats,
    "weekly_coverage_extremes": {
        "min_week_users_ge90_share_all": int(week_summary.loc[week_summary.n_users_ge1_90_share_all.idxmin(), "week"]),
        "max_week_users_ge90_share_all": int(week_summary.loc[week_summary.n_users_ge1_90_share_all.idxmax(), "week"]),
        "min_week_users_ge90_share_active": int(week_summary.loc[week_summary.n_users_ge1_90_share_active.idxmin(), "week"]),
        "max_week_users_ge90_share_active": int(week_summary.loc[week_summary.n_users_ge1_90_share_active.idxmax(), "week"]),
    },
    "tables": saved_tables,
    "figures": saved_figures,
    "limitations": [
        "Наблюдаются только показанные видео, а не все потенциальные предпочтения",
        "Пропуски активности на отдельных неделях не равны отрицательным реакциям",
        "Популярность и вероятность глубокого просмотра не эквивалентны",
        "Это EDA, пороги глубокого просмотра не являются согласованным таргетом",
        "Официальный отбор активных пользователей и популярных видео ограничивает обобщение на всю VK",
    ],
}
output = OUT / "summary.json"
output.write_text(json.dumps(as_jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print("Проверки выполнены!")
print("CSV:", len(saved_tables), "HTML:", len(saved_figures))
print("Сводка:", output.resolve())
print('Результаты сохранены.')