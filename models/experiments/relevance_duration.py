"""relevance duration experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display, Markdown

ROOT = Path.cwd()
WORK = ROOT / "data" / "work"
OUT = ROOT / "results" / "relevance_duration_thresholds"
TAB = OUT / "tables"
FIG = OUT / "figures"
for folder in (TAB, FIG):
    folder.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (30, 45, 60)
MIN_DEPTH_FOR_ABS = 0.25
MIN_EVENTS_FOR_RATE = 100  # ниже порога частоты реакций показываем, но не интерпретируем
DURATION_EDGES = [0, 15, 30, 60, 90, 120, 150, 180, np.inf]
DURATION_LABELS = ["1–15", "16–30", "31–60", "61–90", "91–120", "121–150", "151–180", ">180"]

TABLES, FIGURES = [], []

def save_table(name, df, purpose):
    fn = TAB / f"{name}.csv"
    df.to_csv(fn, index=False, encoding="utf-8-sig")
    TABLES.append({"filename": fn.name, "description": purpose, "rows": int(len(df))})
    return df

def show_plot(name, fig, purpose):
    fig.update_layout(template="plotly_white", height=460,
                      margin=dict(l=65, r=25, t=70, b=75), hovermode="closest")
    fig.write_html(str(FIG / f"{name}.html"), include_plotlyjs="directory", full_html=True)
    FIGURES.append({"filename": f"{name}.html", "description": purpose})
    display(fig)

def serializable(x):
    if isinstance(x, dict): return {str(k): serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serializable(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    return x

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    margin = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (float(center-margin), float(center+margin))

for p in [WORK / "interactions.parquet", WORK / "items_metadata.parquet"]:
    if not p.exists():
        raise FileNotFoundError(f"Не найден {p}. Запусти ноутбук из корневой папки проекта.")


events = pd.read_parquet(WORK / "interactions.parquet")
items = pd.read_parquet(WORK / "items_metadata.parquet")
cols = ["user_id", "item_id", "timespent", "like", "bookmark", "dislike", "share"]
missing = set(cols) - set(events.columns)
if missing:
    raise ValueError(f"В interactions.parquet отсутствуют поля: {sorted(missing)}")
if not {"item_id", "duration"}.issubset(items.columns):
    raise ValueError("В items_metadata.parquet требуются item_id и duration")
if items.item_id.duplicated().any() or events.duplicated(["user_id", "item_id"]).any():
    raise ValueError("Обнаружены дубли item_id в метаданных или user-item в событиях")

if "week" in events.columns:
    weeks = pd.to_numeric(events["week"], errors="coerce")
    if weeks.isna().any() or not weeks.between(0, 24).all():
        raise ValueError("Ожидались только исторические недели 0–24; проверь содержимое data/work")

df = events[cols].copy()
for col in ["timespent", "like", "bookmark", "dislike", "share"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in ["like", "bookmark", "dislike", "share"]:
    if df[col].isna().any() or not df[col].isin([0,1]).all():
        raise ValueError(f"Некорректная реакция {col}")
    df[col] = df[col].astype(bool)
if df.timespent.isna().any() or (df.timespent < 0).any():
    raise ValueError("Некорректное время просмотра")

df["duration"] = df.item_id.map(items.set_index("item_id")["duration"])
if df.duration.isna().any() or (df.duration <= 0).any():
    raise ValueError("Пропуски или неположительная duration после соединения с метаданными")
df["ratio"] = df.timespent / df.duration
df["duration_group"] = pd.cut(df.duration, DURATION_EDGES, labels=DURATION_LABELS, include_lowest=True)

quality = {
    "events": int(len(df)), "users": int(df.user_id.nunique()),
    "items": int(df.item_id.nunique()), "duration_min": float(df.duration.min()),
    "duration_max": float(df.duration.max()),
    "ratio_over_1": int((df.ratio > 1).sum()),
    "likes": int(df.like.sum()), "bookmarks": int(df.bookmark.sum()),
    "dislikes": int(df.dislike.sum()), "shares": int(df.share.sum()),
    "historical_weeks_only": bool("week" not in events.columns or weeks.between(0,24).all())
}
if quality["events"] != 1_454_216 or quality["users"] != 3000 or quality["items"] != 2000:
    warnings.warn("Размер выборки отличается от изученной ранее (1 454 216 / 3 000 / 2 000). Проверь, что это те же данные.")
save_table("00_quality", pd.DataFrame([quality]), "Размеры, полнота, реакции и неизменность выборки")
display(pd.Series(quality).to_frame("value"))


strong = ((df.ratio >= .80) & (df.timespent >= 5)) | ((df.like | df.bookmark) & (df.timespent >= 3))
moderate = (df.ratio >= .50) & (df.timespent >= 5)
base = np.where(df.dislike, 0, np.where(strong, 2, np.where(moderate, 1, 0))).astype("uint8")

grades = {"baseline": base}
promoted = {}
rows = []
user_rows = []
item_rows = []
all_users = pd.Index(df.user_id.unique(), name="user_id")
all_items = pd.Index(df.item_id.unique(), name="item_id")

def counts_per_id(key, values, all_ids):
    return (pd.Series(values, index=df[key]).groupby(level=0).sum()
            .reindex(all_ids, fill_value=0).to_numpy())

for tau in (None, *THRESHOLDS):
    name = "baseline" if tau is None else f"{tau}s"
    if tau is None:
        g = base
        shift = np.zeros(len(df), dtype=bool)
    else:
        abs_rule = (df.timespent >= tau) & (df.ratio >= MIN_DEPTH_FOR_ABS)
        g = np.where(df.dislike, 0,
            np.where(strong, 2, np.where(moderate | abs_rule, 1, 0))).astype("uint8")
        shift = (base == 0) & (g == 1)
        if not np.all((g == base) | shift):
            raise AssertionError("Сценарий изменил метки не только 0→1")
        if not np.array_equal(g == 2, base == 2):
            raise AssertionError("Изменились метки сильного интереса")
    grades[name] = g
    promoted[name] = shift
    n0, n1, n2 = (int((g == lev).sum()) for lev in (0,1,2))
    pos_user = counts_per_id("user_id", g > 0, all_users)
    pos_item = counts_per_id("item_id", g > 0, all_items)
    prom_user = counts_per_id("user_id", shift, all_users)
    prom_item = counts_per_id("item_id", shift, all_items)
    rows.append({"scenario": name, "threshold_seconds": tau, "level0": n0, "level1": n1,
                 "level2": n2, "positive_total": n1+n2,
                 "positive_rate": (n1+n2)/len(df),
                 "new_0_to_1": int(shift.sum()), "new_share_of_all": float(shift.mean()),
                 "new_pct_of_baseline_positive": float(100*shift.sum()/max(int((base>0).sum()),1)),
                 "users_with_new_ge1": int((prom_user > 0).sum()),
                 "users_with_new_ge5": int((prom_user >= 5).sum()),
                 "videos_with_new_ge1": int((prom_item > 0).sum()),
                 "videos_with_new_ge5": int((prom_item >= 5).sum()),
                 "users_with_total_positive_ge5": int((pos_user >= 5).sum()),
                 "items_with_total_positive_ge5": int((pos_item >= 5).sum()),
                 "median_new_per_user": float(np.median(prom_user)),
                 "median_new_per_item": float(np.median(prom_item))})
    user_rows.extend({"scenario":name, "user_id":uid,
                      "n_positive":int(p), "n_new":int(add)}
                     for uid, p, add in zip(all_users, pos_user, prom_user))
    item_rows.extend({"scenario":name, "item_id":iid,
                      "n_positive":int(p), "n_new":int(add)}
                     for iid, p, add in zip(all_items, pos_item, prom_item))

scenarios = save_table("01_scenarios", pd.DataFrame(rows), "Сравнение всех уровней и покрытия 0→1")
save_table("01_user_coverage", pd.DataFrame(user_rows), "Индивидуальные счётчики положительных и новых событий")
save_table("01_item_coverage", pd.DataFrame(item_rows), "Счётчики положительных и новых событий для видео")
assert int((base == 0).sum()) + int((base == 1).sum()) + int((base == 2).sum()) == len(df)
assert sum(int((grades[n] == 2).sum()) for n in grades) == len(grades) * int((base == 2).sum())
display(scenarios)


by_duration_rows = []
for name, g in grades.items():
    tmp = pd.DataFrame({"duration_group":df.duration_group, "rel":g, "new":promoted[name]})
    agg = (tmp.groupby("duration_group", observed=False)
           .agg(n_events=("rel","size"), level0=("rel",lambda x: int((x==0).sum())),
                level1=("rel",lambda x: int((x==1).sum())),
                level2=("rel",lambda x: int((x==2).sum())),
                newly_positive=("new","sum")).reset_index())
    for _, x in agg.iterrows():
        n = int(x.n_events)
        by_duration_rows.append({"scenario":name, "duration_group":str(x.duration_group),
                                 "n_events":n, "level0":int(x.level0),"level1":int(x.level1),
                                 "level2":int(x.level2),"new_0_to_1":int(x.newly_positive),
                                 "share_new_within_group":float(x.newly_positive/n) if n else 0.,
                                 "share_positive_within_group":float((x.level1+x.level2)/n) if n else 0.})
by_duration = save_table("02_by_duration", pd.DataFrame(by_duration_rows),
                         "Новые 0→1 и суммарные позитивные события по длительности")
display(by_duration[by_duration.scenario != "baseline"])

fig=go.Figure()
for name,color in [("30s","#4A8AB7"),("45s","#CE8B3A"),("60s","#745FB1")]:
    subset=by_duration[by_duration.scenario == name]
    fig.add_bar(name=name, x=subset.duration_group, y=subset.new_0_to_1,
                marker_color=color,
                customdata=np.stack([subset.n_events,subset.share_new_within_group*100],axis=1),
                hovertemplate="Длительность: %{x} с<br>Новых меток: %{y:,}<br>Всего событий: %{customdata[0]:,.0f}<br>Доля в группе: %{customdata[1]:.2f}%<extra>%{fullData.name}</extra>")
fig.update_layout(title="Каким длительностям добавляются положительные метки?",
                  xaxis_title="Длительность видео (секунды)",yaxis_title="Новых меток 0→1", barmode="group")
show_plot("01_promoted_by_duration",fig,"Абсолютное число новых положительных меток по длительности")

fig=go.Figure()
for name,color in [("baseline","#788792"),("30s","#4A8AB7"),("45s","#CE8B3A"),("60s","#745FB1")]:
    s=by_duration[by_duration.scenario==name]
    fig.add_trace(go.Scatter(x=s.duration_group,y=s.share_positive_within_group*100,
                             mode="lines+markers",name=name,line=dict(color=color,width=2),
                             customdata=np.array(s.n_events),
                             hovertemplate="Длительность: %{x} с<br>Положительных: %{y:.2f}%<br>Событий: %{customdata}<extra>%{fullData.name}</extra>"))
fig.update_layout(title="Как меняется доля положительных по длительности?",
                  xaxis_title="Длительность видео (секунды)", yaxis_title="Доля событий уровня 1 или 2, %")
show_plot("02_positive_share_by_duration",fig,"Доля rel≥1 по интервалам длительности")


marker_rows=[]
for tau in THRESHOLDS:
    mask = (df.ratio < .5) & (df.duration > 2*tau)
    sub = df.loc[mask, ["duration_group", "timespent", "ratio", "like", "bookmark", "dislike", "share"]].copy()
    sub["eligible"] = (sub.timespent >= tau) & (sub.ratio >= MIN_DEPTH_FOR_ABS)
    sub["positive_reaction"] = sub.like | sub.bookmark
    agg = (sub.groupby(["duration_group","eligible"],observed=True)
           .agg(n_events=("like","size"),likes=("like","sum"),
                bookmarks=("bookmark","sum"),
                positive_reactions=("positive_reaction","sum"),
                dislikes=("dislike","sum"), shares=("share","sum")).reset_index())
    for _,x in agg.iterrows():
        n=int(x.n_events); likes=int(x.likes); reactions=int(x.positive_reactions)
        lo, hi=wilson(likes,n)
        marker_rows.append({"threshold_seconds":tau, "duration_group":str(x.duration_group),
                            "part":"long_partial" if x.eligible else "short_partial",
                            "n_events":n,"likes":likes, "bookmarks":int(x.bookmarks),
                            "positive_reactions":reactions,"dislikes":int(x.dislikes), "shares":int(x.shares),
                            "like_rate":likes/n if n else np.nan,
                            "like_rate_ci_low":lo,"like_rate_ci_high":hi,
                            "positive_reaction_rate":reactions/n if n else np.nan,
                            "enough_events_for_description":bool(n>=MIN_EVENTS_FOR_RATE)})
markers = save_table("03_reactions_shallow", pd.DataFrame(marker_rows),
                     "Маркерные реакции среди ВСЕХ частичных просмотров, стратификация по длительности")

compare=[]
for (tau, group), g in markers.groupby(["threshold_seconds","duration_group"], sort=False):
    if set(g.part)=={"long_partial","short_partial"}:
        high=g[g.part=="long_partial"].iloc[0]; low=g[g.part=="short_partial"].iloc[0]
        compare.append({"threshold_seconds":int(tau), "duration_group":group,
                        "n_long_partial":int(high.n_events), "n_short_partial":int(low.n_events),
                        "n_likes_long":int(high.likes), "n_likes_short":int(low.likes),
                        "share_rate_long_pct":float(high.shares/high.n_events*100),
                        "share_rate_short_pct":float(low.shares/low.n_events*100),
                        "like_rate_long_pct":float(high.like_rate*100),
                        "like_rate_short_pct":float(low.like_rate*100),
                        "difference_pp":float((high.like_rate-low.like_rate)*100),
                        "rate_ratio":float(high.like_rate/low.like_rate) if low.like_rate else np.nan,
                        "supports_comparison":bool(min(high.n_events,low.n_events)>=MIN_EVENTS_FOR_RATE)})
comparison=save_table("03_reaction_comparison", pd.DataFrame(compare),
                      "Сопоставление частичных просмотров с порогом и без порога внутри групп длительности")
display(comparison)

# График для основного кандидата 45 секунд, по каждому интервалу длительности.
focus = markers[markers.threshold_seconds==45]
fig=go.Figure()
for label,color in [("short_partial","#75899D"),("long_partial","#D18B3A")]:
    v=focus[focus.part==label]
    fig.add_bar(x=v.duration_group,y=v.like_rate*100,
                error_y=dict(type="data",array=(v.like_rate_ci_high-v.like_rate)*100,
                             arrayminus=(v.like_rate-v.like_rate_ci_low)*100,visible=True),
                name=("Ниже порога" if label=="short_partial" else "Достигнут порог"),
                marker_color=color,customdata=np.stack([v.n_events,v.likes],axis=1),
                hovertemplate="Видео: %{x} с<br>Like-rate: %{y:.3f}%<br>N=%{customdata[0]:,.0f}<br>Лайков=%{customdata[1]:,.0f}<extra>%{fullData.name}</extra>")
fig.update_layout(title="45 секунд: как различается частота лайков среди частичных просмотров?",
                  xaxis_title="Длительность видео (секунды)", yaxis_title="Like-rate, %",barmode="group")
show_plot("03_like_rate_by_partial_watch",fig,
          "Сравнение like-rate с 95%-м интервалом Вилсона (описательно, не причинно)")


floor_rows=[]
for tau in THRESHOLDS:
    no_floor = (base == 0) & (df.timespent >= tau) & (~df.dislike)
    with_floor = promoted[f"{tau}s"]
    excluded = no_floor & ~with_floor
    floor_rows.append({"threshold_seconds":tau, "new_without_depth_floor":int(no_floor.sum()),
                       "new_with_depth_floor":int(with_floor.sum()),
                       "blocked_by_25pct_floor":int(excluded.sum()),
                       "min_depth_blocked":float(df.loc[excluded,"ratio"].min()) if excluded.any() else np.nan,
                       "median_depth_blocked":float(df.loc[excluded,"ratio"].median()) if excluded.any() else np.nan})
floor_table=save_table("04_depth_floor",pd.DataFrame(floor_rows),
                       "Что меняет ограничение дополнительного времени глубиной ≥25%")
display(floor_table)

user_added = pd.DataFrame(user_rows)
fig=go.Figure()
for name,color in [("30s","#4A8AB7"),("45s","#CE8B3A"),("60s","#745FB1")]:
    count = np.sort(user_added.loc[user_added.scenario==name,"n_new"].to_numpy())
    xs = np.r_[count, count[-1]] if len(count) else []
    ys = np.r_[np.arange(1,len(count)+1)/len(count),1] if len(count) else []
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color=color,width=2),name=name))
fig.update_layout(title="Сколько новых положительных меток достаётся каждому пользователю?",
                  xaxis_title="Количество новых меток 0→1 на пользователя",yaxis_title="Доля пользователей (ECDF)")
show_plot("04_promotions_per_user_ecdf",fig,
          "Покрытие пользователей новыми умеренно релевантными событиями")


# Проверки сохранения сумм для разных срезов.
for name in grades:
    g=grades[name]
    assert int((g==0).sum() + (g==1).sum() + (g==2).sum()) == len(df)
    s=by_duration[by_duration.scenario==name]
    assert int(s.n_events.sum()) == len(df)
    assert int(s.new_0_to_1.sum()) == int(promoted[name].sum())
    assert int(s.level1.sum()) == int((g==1).sum())
    assert int(s.level2.sum()) == int((g==2).sum())
    assert int(user_added.loc[user_added.scenario==name,"n_new"].sum())==int(promoted[name].sum())

summary={
    "project":"VK-LSVD: absolute watch time for moderate relevance",
    "scope":"existing working sample, historical weeks 0–24 only",
    "data_quality":quality,
    "definitions":{
        "grade_0":"dislike veto or neither grade 1 nor 2",
        "grade_2":"(ratio >= 0.80 and seconds >= 5) or ((like or bookmark) and seconds >= 3), unless dislike",
        "baseline_grade_1":"ratio >= 0.50 and seconds >= 5, unless grade 2 or dislike",
        "extra_grade_1":"seconds >= threshold and ratio >= 0.25, unless grade 2 or dislike",
        "thresholds_seconds":list(THRESHOLDS),
        "like_rate_diagnostic":"All shallow views ratio < 0.5 in durations > 2*threshold, stratified by video duration; includes events with explicit reactions, not independent ground truth"
    },
    "scenarios":scenarios.to_dict(orient="records"),
    "by_duration":by_duration.to_dict(orient="records"),
    "reaction_comparisons":comparison.to_dict(orient="records"),
    "depth_floor":floor_table.to_dict(orient="records"),
    "tables":TABLES, "figures":FIGURES,
    "warnings":[
        "Like-rate comparisons are correlational and may be confounded by user propensity, video mix, and exposure policy.",
        "Liking is also part of grade 2; marker-rate analysis must not be restricted to promoted events.",
        "No validation/test rows were inspected; decision is only a pre-model heuristic."
    ]
}
(OUT / "summary.json").write_text(json.dumps(serializable(summary),ensure_ascii=False,indent=2),encoding="utf-8")
lines=["# Как читать результаты порогов", "", "1. `tables/01_scenarios.csv` — число изменённых меток и охват пользователей/видео.",
       "2. `tables/02_by_duration.csv` — добавленные метки и доли по длительности.",
       "3. `tables/03_reaction_comparison.csv` — описательная проверка лайков при частичных просмотрах, по длительности.",
       "4. `tables/04_depth_floor.csv` — влияние дополнительного ограничения 25%.",
       "5. `summary.json` — те же показатели плюс параметры и методологические предупреждения.",
       "", "Отдельные графики находятся в figures/. Число новых меток само по себе НЕ определяет лучший порог."]
(OUT/"README_results.md").write_text("\n".join(lines),encoding="utf-8")
print("Все проверки согласованности пройдены.")
print(f"Таблиц: {len(TABLES)}, графиков: {len(FIGURES)}. Результаты: {OUT.resolve()}")
