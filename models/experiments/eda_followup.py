"""eda followup experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

ROOT = Path.cwd()
WORK = ROOT / "data" / "work"
OUT = ROOT / "results" / "eda_followup"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
for p in (TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

required_files = [WORK / name for name in
                  ("interactions.parquet", "users_metadata.parquet", "items_metadata.parquet")]
missing_files = [str(p) for p in required_files if not p.is_file()]
if missing_files:
    raise FileNotFoundError(
        "Рабочая выборка не найдена. Запусти Jupyter из корня первого проекта; "
        "эти файлы должны существовать: " + ", ".join(missing_files)
    )

REACTS = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")
RATIO_LABELS = ["0–10%", "10–25%", "25–50%", "50–75%", "75–100%", "100–150%", "≥150%"]
RATIO_EDGES = [0, .10, .25, .50, .75, 1.0, 1.50, float("inf")]
DUR_LABELS = ["1–15с", "16–30с", "31–60с", "61–120с", "121–180с", ">180с"]
DUR_EDGES = [0, 15, 30, 60, 120, 180, float("inf")]
THRESHOLDS = [0.50, 0.80, 0.90, 1.00]
MIN_CELL_N = 500 # Только порог показа/интерпретации: строки не удаляются из расчётов.

TABLE_INDEX = []
FIGURE_INDEX = []

def save_table(name, table, description):
    if isinstance(table, pd.Series):
        table = table.reset_index()
    p = TABLES / f"{name}.csv"
    table.to_csv(p, index=False, encoding="utf-8-sig")
    TABLE_INDEX.append({"file": p.name, "description": description, "rows": len(table)})
    return table

def show_fig(name, fig, description):
    fig.update_layout(template="plotly_white", height=480,
                      margin=dict(l=60, r=35, t=75, b=75), hovermode="closest")
    fig.write_html(str(FIGURES / f"{name}.html"), include_plotlyjs="directory", full_html=True)
    FIGURE_INDEX.append({"file": f"{name}.html", "description": description})
    display(fig)

def round_native(value):
    if isinstance(value, dict):
        return {str(k): round_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value

def pct(x):
    return f"{x * 100:.2f}%" if pd.notna(x) else "н/д"

def wilson_interval(k, n, z=1.96):
    """Приближённый 95%-й интервал для доли (описательная неопределённость)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    good = n > 0
    p = np.divide(k, n, out=np.zeros_like(k), where=good)
    d = 1 + z*z/np.maximum(n, 1)
    center = (p + z*z/(2*np.maximum(n, 1)))/d
    half = z*np.sqrt((p*(1-p) + z*z/(4*np.maximum(n, 1)))/np.maximum(n, 1))/d
    return np.where(good, np.maximum(0, center-half), np.nan), np.where(good, np.minimum(1, center+half), np.nan)

print("Проект:", ROOT)
print("Вывод результатов:", OUT)


events = pd.read_parquet(WORK / "interactions.parquet")
users = pd.read_parquet(WORK / "users_metadata.parquet")
items = pd.read_parquet(WORK / "items_metadata.parquet")

required_events = {"user_id", "item_id", "week", "timespent", *REACTS}
required_items = {"item_id", "duration"}
for label, missing in (("interactions", required_events - set(events.columns)),
                       ("items_metadata", required_items - set(items.columns))):
    if missing:
        raise ValueError(f"Неизвестная структура {label}; недостающие колонки: {sorted(missing)}")
if items.item_id.duplicated().any() or users.user_id.duplicated().any():
    raise ValueError("Повторяются ID метаданных: соединение неоднозначно")
if events.duplicated(["user_id", "item_id"]).any():
    raise ValueError("Обнаружены повторяющиеся user-item пары: требуется отдельно выяснить причину")

# Для памяти оставляем только необходимые столбцы, метаданные присоединяем по ключу.
cols = ["user_id", "item_id", "week", "timespent", *REACTS]
df = events.loc[:, cols].copy()
for col in REACTS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[col].isna().any() or not df[col].isin([0, 1]).all():
        raise ValueError(f"Реакция '{col}' имеет пропуски либо значения вне {{0,1}}")
df["duration"] = df["item_id"].map(items.set_index("item_id")["duration"])
df["ratio"] = np.where(
    df["duration"].notna() & (df["duration"] > 0),
    df["timespent"] / df["duration"].replace(0, np.nan), np.nan
)
df["duration_group"] = pd.cut(df["duration"].where(df["duration"] > 0), bins=DUR_EDGES,
                               labels=DUR_LABELS, include_lowest=True)
df["ratio_group"] = pd.cut(df["ratio"], bins=RATIO_EDGES,
                            labels=RATIO_LABELS, right=False, include_lowest=True)

n_users = int(df.user_id.nunique())
n_items = int(df.item_id.nunique())
source_weeks = sorted(df.week.dropna().unique().tolist())
quality = {
    "events": int(len(df)), "users": n_users, "items": n_items,
    "weeks": [int(w) for w in source_weeks],
    "density": float(len(df) / (n_users * n_items)),
    "valid_ratio_rows": int(df.ratio.notna().sum()),
    "ratio_over_one_rows": int((df.ratio > 1).sum()),
    "missing_duration_rows": int(df.duration.isna().sum()),
    "duration_nonpositive_rows": int((df.duration <= 0).sum()),
    "duplicate_pairs": int(df.duplicated(["user_id", "item_id"]).sum()),
    "assignment_size_ok": bool(100_000 <= len(df) <= 2_000_000 and n_users >= 1000 and n_items >= 1000),
}
assert quality["assignment_size_ok"], "Выборка не удовлетворяет численным требованиям ДЗ"
print(json.dumps(round_native(quality), ensure_ascii=False, indent=2))
save_table("00_data_quality", pd.DataFrame([quality]).drop(columns="weeks"),
           "Базовая проверка размеров, плотности и определения доли просмотра")


all_users = pd.Index(sorted(df.user_id.unique()), name="user_id")
counts = pd.DataFrame({"user_id": all_users})
counts["n_events"] = df.groupby("user_id").size().reindex(all_users, fill_value=0).to_numpy()
counts["likes"] = df.groupby("user_id")["like"].sum().reindex(all_users, fill_value=0).to_numpy()
cover_rows = []

for t in THRESHOLDS:
    label = f"watch_ge_{int(t*100)}"
    matched = df["ratio"].notna() & (df["ratio"] >= t)
    per_user = df.loc[matched].groupby("user_id").size().reindex(all_users, fill_value=0)
    counts[label] = per_user.to_numpy()
    cover_rows.append({
        "signal": f"просмотр ≥{int(t * 100)}%",
        "event_count": int(matched.sum()), "event_share": float(matched.mean()),
        "users_ge_1": int((per_user >= 1).sum()),
        "users_ge_5": int((per_user >= 5).sum()),
        "users_ge_10": int((per_user >= 10).sum()),
        "users_ge_1_share": float((per_user >= 1).mean()),
        "users_ge_5_share": float((per_user >= 5).mean()),
        "median_events_per_user": float(per_user.median()),
        "p10_events_per_user": float(per_user.quantile(.1)),
        "p90_events_per_user": float(per_user.quantile(.9)),
    })

likes_user = counts["likes"]
cover_rows.append({
    "signal": "лайк", "event_count": int(df["like"].sum()),
    "event_share": float(df["like"].mean()),
    "users_ge_1": int((likes_user >= 1).sum()),
    "users_ge_5": int((likes_user >= 5).sum()),
    "users_ge_10": int((likes_user >= 10).sum()),
    "users_ge_1_share": float((likes_user >= 1).mean()),
    "users_ge_5_share": float((likes_user >= 5).mean()),
    "median_events_per_user": float(likes_user.median()),
    "p10_events_per_user": float(likes_user.quantile(.1)),
    "p90_events_per_user": float(likes_user.quantile(.9)),
})
coverage = save_table("01_coverage", pd.DataFrame(cover_rows),
                     "Покрытие сигналов по событиям и пользователям, пороги глубины просмотра")
save_table("01_user_signal_counts", counts, "Количество событий каждого типа на каждого пользователя")
display(coverage.round(4))

fig = go.Figure()
for metric, name in [("users_ge_1_share", "хотя бы 1 событие"),
                     ("users_ge_5_share", "хотя бы 5 событий")]:
    fig.add_trace(go.Bar(x=coverage.signal, y=100*coverage[metric], name=name,
                         customdata=coverage.event_count,
                         hovertemplate="%{x}<br>Пользователей: %{y:.2f}%<br>Всего событий: %{customdata:,}<extra></extra>"))
fig.update_layout(barmode="group", title="Какой доле пользователей доступен каждый сигнал?",
                  yaxis_title="Доля пользователей, %", xaxis_title="Сигнал")
show_fig("01_signal_coverage", fig, "Доля пользователей с ≥1 и ≥5 событиями каждого типа")

fig = go.Figure()
for col, label in [("watch_ge_50", "≥50%"), ("watch_ge_80", "≥80%"),
                   ("watch_ge_90", "≥90%"), ("watch_ge_100", "≥100%"), ("likes", "лайки")]:
    vals = np.sort(counts[col].to_numpy())
    fig.add_trace(go.Scatter(x=vals, y=np.arange(1, len(vals)+1)/len(vals),
                             mode="lines", name=label))
fig.update_layout(title="ECDF: число подходящих событий на пользователя",
                  xaxis_title="Количество событий", yaxis_title="Накопленная доля пользователей")
show_fig("01_signal_count_ecdf", fig, "Распределение числа просмотров заданной глубины и лайков на пользователя")


ratio_dur = (df.dropna(subset=["duration_group", "ratio_group"])
             .groupby(["duration_group", "ratio_group"], observed=False)
             .agg(n=("like", "size"), like_count=("like", "sum"))
             .reset_index())
ratio_dur["like_rate"] = np.where(ratio_dur.n>0, ratio_dur.like_count/ratio_dur.n, np.nan)
ratio_dur["low95"], ratio_dur["high95"] = wilson_interval(ratio_dur.like_count, ratio_dur.n)
ratio_dur = save_table("02_like_rate_by_duration_and_ratio", ratio_dur,
                       "Лайки по совместным группам длительности и глубины просмотра, с численностями и интервалами")

dur_summary = (df.dropna(subset=["duration_group"])
               .groupby("duration_group", observed=False)
               .agg(n=("user_id", "size"), like_count=("like", "sum"),
                    median_ratio=("ratio", "median"), mean_ratio=("ratio", "mean"))
               .reset_index())
dur_summary["like_rate"] = dur_summary.like_count / dur_summary.n.replace(0, np.nan)
save_table("02_duration_summary", dur_summary,
           "Общие частоты лайков и отношение просмотра по группам длительности")
display(dur_summary)

pivot_rate = ratio_dur.pivot(index="duration_group", columns="ratio_group", values="like_rate")
pivot_n = ratio_dur.pivot(index="duration_group", columns="ratio_group", values="n")
pivot_rate = pivot_rate.reindex(index=DUR_LABELS, columns=RATIO_LABELS)
pivot_n = pivot_n.reindex(index=DUR_LABELS, columns=RATIO_LABELS)
values = np.where(pivot_n.to_numpy() >= MIN_CELL_N, pivot_rate.to_numpy() * 100, np.nan)
fig = go.Figure(go.Heatmap(
    x=RATIO_LABELS, y=DUR_LABELS, z=values,
    customdata=pivot_n.to_numpy(), colorscale="Blues", colorbar={"title":"Like-rate, %"},
    hovertemplate="Длительность: %{y}<br>Просмотр: %{x}<br>Лайк: %{z:.3f}%<br>N: %{customdata:,}<extra></extra>"
))
fig.update_layout(title=f"Like-rate по доле просмотра и длительности (показаны n≥{MIN_CELL_N})",
                  xaxis_title="Доля просмотра", yaxis_title="Длительность видео")
show_fig("02_like_duration_ratio_heatmap", fig, "Like-rate по глубине просмотра и длительности видео")

fig = go.Figure()
for d in DUR_LABELS:
    sub = ratio_dur[(ratio_dur.duration_group == d) & (ratio_dur.n >= MIN_CELL_N)]
    fig.add_trace(go.Scatter(x=sub.ratio_group.astype(str), y=100*sub.like_rate,
                             mode="lines+markers", name=d,
                             customdata=sub[["n", "like_count"]],
                             hovertemplate="Доля: %{x}<br>Like-rate: %{y:.3f}%<br>N=%{customdata[0]:,}<br>Лайков=%{customdata[1]:,}<extra></extra>"))
fig.update_layout(title="Как меняется like-rate с глубиной просмотра внутри групп длительности",
                  xaxis_title="Доля просмотра", yaxis_title="Like-rate, %")
show_fig("02_like_duration_ratio_lines", fig, "Профили like-rate по группам длительности")


activity = df.groupby("user_id").size().rename("n_events").reset_index()
activity = activity.sort_values(["n_events", "user_id"]).reset_index(drop=True)
q_labels = ["Q1 (низкая)", "Q2", "Q3", "Q4 (высокая)"]
activity["activity_group"] = pd.qcut(np.arange(len(activity)), 4, labels=q_labels)
activity["activity_group"] = pd.Categorical(activity.activity_group,
                                            categories=q_labels, ordered=True)
user_rate = df.groupby("user_id")["like"].mean().rename("individual_like_rate")
activity = activity.merge(user_rate, on="user_id", validate="one_to_one")
df["activity_group"] = df["user_id"].map(activity.set_index("user_id")["activity_group"])

per_user_summary = (activity.groupby("activity_group", observed=False)
                    .agg(n_users=("user_id", "size"), activity_min=("n_events", "min"),
                         activity_median=("n_events", "median"),
                         activity_max=("n_events", "max"),
                         mean_individual_like_rate=("individual_like_rate", "mean"))
                    .reset_index())
group_events = (df.groupby("activity_group", observed=False)
                .agg(n=("user_id", "size"), like_count=("like", "sum"),
                     dislike_count=("dislike", "sum"), bookmark_count=("bookmark", "sum"),
                     share_count=("share", "sum"),
                     median_ratio=("ratio", "median"), median_duration=("duration", "median"))
                .reset_index())
group_events["share_ratio_ge_80"] = df.groupby("activity_group", observed=False)["ratio"].apply(
    lambda s: (s >= .8).mean()).to_numpy()
for reaction in ["like", "dislike", "bookmark", "share"]:
    group_events[f"{reaction}_rate"] = group_events[f"{reaction}_count"] / group_events.n
activity_summary = save_table("03_activity_quartiles", per_user_summary.merge(group_events,
    on="activity_group", validate="one_to_one"),
    "Характеристики квартилей активности, доли реакций, глубина просмотра и индивидуальные частоты")
save_table("03_user_activity_assignment", activity,
           "Число событий, группа активности и like-rate каждого пользователя")
display(activity_summary.round(5))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Доля лайков", "Просмотры ≥80%"))
fig.add_trace(go.Bar(x=activity_summary.activity_group, y=100*activity_summary.like_rate,
                     name="Like-rate", marker_color="#3B82B6"), row=1,col=1)
fig.add_trace(go.Bar(x=activity_summary.activity_group, y=100*activity_summary.share_ratio_ge_80,
                     name="≥80%", marker_color="#7C9B6C"), row=1,col=2)
fig.update_yaxes(title_text="% событий", row=1,col=1)
fig.update_yaxes(title_text="% событий", row=1,col=2)
fig.update_layout(title="Явные реакции и глубина просмотра у разных групп активности", showlegend=False)
show_fig("03_activity_like_and_watch", fig, "Like-rate и доля глубоких просмотров по квартилям активности")

qr = (df.dropna(subset=["ratio_group"])
      .groupby(["activity_group", "ratio_group"], observed=False)
      .agg(n=("like", "size"), like_count=("like", "sum"))
      .reset_index())
qr["like_rate"] = qr.like_count / qr.n.replace(0, np.nan)
qr = save_table("03_like_rate_within_ratio_bands", qr,
                "Like-rate по квартилям активности внутри одинаковых диапазонов глубины")
fig = go.Figure()
for q in q_labels:
    s = qr[(qr.activity_group == q) & (qr.n >= MIN_CELL_N)]
    fig.add_trace(go.Scatter(x=s.ratio_group.astype(str), y=100*s.like_rate,
                             name=q, mode="lines+markers", customdata=s.n,
                             hovertemplate="Глубина: %{x}<br>Like-rate: %{y:.3f}%<br>N=%{customdata:,}<extra></extra>"))
fig.update_layout(title="Like-rate при одинаковых интервалах глубины просмотра",
                  xaxis_title="Доля просмотра", yaxis_title="Like-rate, %")
show_fig("03_like_within_watch_bands", fig, "Сравнение like-rate между группами активности при одинаковой глубине")


weekly = (df.groupby("week")
          .agg(n=("user_id", "size"), n_users=("user_id", "nunique"),
               n_items=("item_id", "nunique"), median_duration=("duration", "median"),
               median_ratio=("ratio", "median"),
               like_count=("like", "sum"))
          .reset_index().sort_values("week"))
weekly["like_rate"] = weekly.like_count / weekly.n
weekly["share_watch_ge_80"] = (df.assign(high_watch=(df.ratio >= .80))
                              .groupby("week")["high_watch"].mean().reindex(weekly.week).to_numpy())
weekly["events_pct_change"] = weekly.n.pct_change()
weekly = save_table("04_weekly_summary", weekly,
                    "Основные метрики каждой недели и недельные изменения")

wd = (df.dropna(subset=["duration_group"])
      .groupby(["week", "duration_group"], observed=False)
      .agg(n=("user_id", "size"), count_ge_80=("ratio", lambda s: int((s>=.8).sum())),
           likes=("like", "sum"))
      .reset_index())
wd["share_of_week"] = wd.n / wd.groupby("week").n.transform("sum").replace(0, np.nan)
wd["share_ge_80"] = wd.count_ge_80 / wd.n.replace(0, np.nan)
wd = save_table("04_weekly_duration_mix", wd,
                "Состав просмотров по длительности для каждой недели; глубина внутри групп")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    subplot_titles=("Число взаимодействий", "Активных видео / пользователей"),
                    vertical_spacing=.13)
fig.add_trace(go.Scatter(x=weekly.week, y=weekly.n, mode="lines+markers", name="События"), row=1,col=1)
fig.add_trace(go.Scatter(x=weekly.week, y=weekly.n_items, mode="lines+markers", name="Видео"), row=2,col=1)
fig.add_trace(go.Scatter(x=weekly.week, y=weekly.n_users, mode="lines+markers", name="Пользователи"), row=2,col=1)
for w in (9, 18, 19):
    fig.add_vline(x=w, line_dash="dash", line_color="grey", opacity=.5)
fig.update_xaxes(title_text="Неделя", dtick=1, row=2,col=1)
fig.update_layout(title="События и доступные сущности по неделям", height=620)
show_fig("04_weekly_volume", fig, "Объёмы, уникальные видео и пользователи по неделям")

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Медианная длительность", "Доля просмотров ≥80%"))
fig.add_trace(go.Scatter(x=weekly.week, y=weekly.median_duration, mode="lines+markers",
                         name="Длительность"), row=1,col=1)
fig.add_trace(go.Scatter(x=weekly.week, y=100*weekly.share_watch_ge_80,
                         mode="lines+markers", name="≥80%"), row=1,col=2)
fig.update_xaxes(title_text="Неделя", dtick=2)
fig.update_yaxes(title_text="Секунды", row=1,col=1)
fig.update_yaxes(title_text="% взаимодействий", row=1,col=2)
fig.update_layout(title="Как меняется длительность и глубина просмотра")
show_fig("04_weekly_duration_watch", fig, "Длительность и глубокие просмотры по неделям")

mix = wd.pivot(index="week", columns="duration_group", values="share_of_week").reindex(columns=DUR_LABELS)
fig = go.Figure()
for label in DUR_LABELS:
    fig.add_trace(go.Bar(x=mix.index, y=100*mix[label].fillna(0), name=label))
fig.update_layout(barmode="stack", title="Как меняется состав роликов по длительности",
                  xaxis_title="Неделя", yaxis_title="Доля событий, %", legend_title="Длительность")
show_fig("04_weekly_duration_mix", fig, "Состав показов по группам длительности")

focus = weekly[weekly.week.isin([8,9,10,17,18,19,20])].copy()
source_ref = ROOT / "results" / "eda" / "source_vs_sample_weekly.csv"
if source_ref.exists():
    src = pd.read_csv(source_ref)
    expected = {"week", "source_interactions", "sample_interactions"}
    if expected.issubset(src.columns):
        src["sample_fraction"] = src.sample_interactions / src.source_interactions.replace(0, np.nan)
        save_table("04_source_sample_weekly", src,
                   "Доля нашей подвыборки от официальной по неделям (если исходные агрегаты доступны)")
        focus = focus.merge(src[["week", "sample_fraction", "source_interactions"]],
                            on="week", how="left", validate="one_to_one")
        fig = go.Figure(go.Scatter(x=src.week, y=100*src.sample_fraction, mode="lines+markers"))
        fig.update_layout(title="Как менялась доля рабочей выборки в официальной по неделям",
                          xaxis_title="Неделя", yaxis_title="Доля событий, %")
        show_fig("04_sample_fraction_by_week", fig, "Проверка изменения доли отобранных событий в официальном источнике")
focus = save_table("04_focus_weeks", focus,
                   "Показатели недель 8–10 и 17–20 для проверки аномалий")
display(focus.round(4))


# Простые описательные сравнения — не статистические тесты и не причинные эффекты.
summary = {
    "project": "VK-LSVD EDA follow-up (historical weeks only)",
    "quality": quality,
    "configuration": {
        "signal_thresholds": THRESHOLDS,
        "ratio_bins_left_closed": True,
        "min_cell_n_for_plot": MIN_CELL_N,
        "no_new_split": True,
        "no_model_training": True,
        "no_target_engineering": True,
    },
    "coverage": coverage.to_dict(orient="records"),
    "activity_quartiles": activity_summary.to_dict(orient="records"),
    "selected_weeks": focus.to_dict(orient="records"),
    "limitations": [
        "Выборка активных пользователей и популярных объектов: не репрезентативна для всей аудитории VK",
        "Реакции наблюдаются только для показанных видео; причинные эффекты не идентифицируются",
        "Номер недели не равен точному timestamp",
        "Timespent/duration > 1 возможно; данные не обрезались",
        "Пороги глубины используются только для EDA, не являются утверждённым таргетом",
    ],
    "tables": TABLE_INDEX,
    "figures": FIGURE_INDEX,
}
(OUT / "summary.json").write_text(
    json.dumps(round_native(summary), ensure_ascii=False, indent=2, allow_nan=False),
    encoding="utf-8"
)
print("Готово.")
print("CSV-таблиц:", len(TABLE_INDEX), "HTML-графиков:", len(FIGURE_INDEX))
print("Папка:", OUT.resolve())
print("Сводка:", (OUT / "summary.json").resolve())
