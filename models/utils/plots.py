"""Plotly charts for the seven approved EDA blocks.
Each chart has a corresponding saved CSV aggregate in results/eda/distributions/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SECTIONS = [
    "01_quality", "02_activity", "03_watch", "04_reactions",
    "05_users", "06_metadata", "07_time",
]


def finish(fig: go.Figure, title: str, x: str = "", y: str = "") -> go.Figure:
    fig.update_layout(title={"text": title, "x": 0.02},
                      template="plotly_white", height=460,
                      margin=dict(l=55, r=25, t=85, b=65),
                      legend_title_text="", hovermode="closest")
    if x:
        fig.update_xaxes(title_text=x)
    if y:
        fig.update_yaxes(title_text=y)
    return fig


def bars(x, y, title, xlabel, ylabel):
    fig = go.Figure(go.Bar(x=x, y=y))
    return finish(fig, title, xlabel, ylabel)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values))
    return values, np.arange(1, len(values) + 1) / len(values)


def lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.sort(np.asarray(values, dtype=float))
    c = np.r_[0.0, np.cumsum(vals)]
    return np.arange(len(c)) / len(vals), c / max(c[-1], 1)


def create_figures(t: dict[str, pd.DataFrame], checks: dict,
                   source: dict[str, pd.DataFrame]) -> dict[str, list[tuple[str, go.Figure]]]:
    f = {x: [] for x in SECTIONS}

    def add(section, name, fig):
        f[section].append((name, fig))

    miss = t["missingness"].sort_values("fraction", ascending=True)
    fig = go.Figure(go.Bar(x=100 * miss.fraction, y=miss.column, orientation="h"))
    add("01_quality", "missingness", finish(fig, "Пропуски по колонкам", "Доля пропусков, %", "Колонка"))
    fig = go.Figure(go.Bar(x=["Дубли user–item", "Лайк + дизлайк", "Timespent = 255", "Duration = 0", "Ratio > 1"],
                           y=[checks["duplicate_user_item_pairs"], checks["like_and_dislike"],
                              checks["timespent_255"], checks["duration_zero_rows"],
                              checks["watch_ratio_over_1"]]))
    add("01_quality", "anomalies", finish(fig, "Проверочные счётчики (не все значения — ошибки)",
                                           "Проверка", "Число строк"))

    ua = t["user_activity"]
    ia = t["item_popularity"]
    fig = go.Figure()
    for name, d in [("Пользователи", ua), ("Видео", ia)]:
        x, y = ecdf(d.n_interactions)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_xaxes(type="log")
    add("02_activity", "activity_ecdf", finish(fig, "Распределение числа взаимодействий (ECDF, лог-X)",
                                                 "События на сущность (log)", "Накопленная доля"))
    fig = go.Figure()
    for name, d in [("Пользователи", ua), ("Видео", ia)]:
        sorted_activity = np.sort(d.n_interactions.to_numpy())[::-1]
        fig.add_trace(go.Scatter(x=np.arange(1, len(sorted_activity)+1), y=sorted_activity,
                                 mode="lines", name=name))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    add("02_activity", "rank_frequency", finish(fig, "Активность / популярность по рангу (log–log)",
                                                  "Ранг (log)", "Число взаимодействий (log)"))
    fig = go.Figure()
    for name, d in [("Пользователи", ua), ("Видео", ia)]:
        x, y = lorenz(d.n_interactions)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Равномерность",
                             line=dict(dash="dash")))
    add("02_activity", "lorenz", finish(fig, "Кривые Лоренца для взаимодействий",
                                         "Доля сущностей (от менее активных)", "Доля взаимодействий"))
    matrix = t["matrix_density_grid"].set_index("user_activity_bin")
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=matrix.columns, y=matrix.index,
                               colorbar=dict(title="Плотность"), zmin=0,
                               hovertemplate="Группа пользователей: %{y}<br>Группа видео: %{x}<br>Доля пар: %{z:.3f}<extra></extra>"))
    add("02_activity", "matrix_density", finish(fig, "Локальная плотность матрицы (группы по активности)",
                                                  "Видео: от популярных к редким", "Пользователи: от активных к редким"))
    if "source_user_activity" in source and "source_item_popularity" in source:
        fig = go.Figure()
        for src, work, label in [(source["source_user_activity"], ua, "пользователи"),
                                 (source["source_item_popularity"], ia, "видео")]:
            for d, suffix in [(src, "официальная"), (work, "рабочая")]:
                x, y = ecdf(d.n_interactions)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{label}: {suffix}"))
        fig.update_xaxes(type="log")
        add("02_activity", "source_vs_sample", finish(fig,
          "Распределения до/после отбора (активность пересчитана на подвыборке)",
          "Число событий (log)", "ECDF"))

    time = t["timespent_hist"]
    add("03_watch", "timespent", bars(time.left, time["count"], "Время просмотра",
                                      "Timespent, секунды (левая граница интервала)", "Число показов"))
    duration = t["duration_hist"]
    add("03_watch", "duration", bars(duration.left, duration["count"],
                                     "Длительность видео (каждый объект один раз)",
                                     "Duration, секунды (левая граница интервала)", "Число видео"))
    ratio = t["watch_ratio_reactions"]
    add("03_watch", "watch_ratio", bars(ratio.ratio_bin, ratio.n,
               "Доля просмотра (включая значения > 1)", "Timespent / duration", "Число показов"))
    dr = t["by_duration"]
    fig = go.Figure(go.Bar(x=dr.duration_group, y=dr.median_ratio,
                           customdata=dr.n, hovertemplate="Длительность: %{x}<br>Медианная доля: %{y:.2f}<br>N=%{customdata:,}<extra></extra>"))
    add("03_watch", "watch_vs_duration", finish(fig, "Медианная доля просмотра по длительности видео",
                                                  "Длительность, с", "Медианная доля просмотра"))

    cross = t["watch_duration_distribution"].set_index("duration_group")
    fig = go.Figure(go.Heatmap(x=cross.columns, y=cross.index,
                               z=100*cross.to_numpy(), colorbar=dict(title="% в группе"),
                               hovertemplate="Длительность: %{y}<br>Просмотр: %{x}<br>В группе: %{z:.2f}%<extra></extra>"))
    add("03_watch", "watch_duration_heatmap", finish(fig,
        "Распределение доли просмотра в группах длительности (по строкам = 100%)",
        "Доля просмотра", "Длительность видео, с"))

    re = t["reactions"]
    fig = go.Figure(go.Bar(x=re.reaction, y=100 * re.rate,
                           customdata=re["count"], hovertemplate="%{x}<br>%{y:.3f}%<br>N=%{customdata:,}<extra></extra>"))
    add("04_reactions", "reaction_frequency", finish(fig, "Частота реакций", "Реакция", "Доля всех показов, %"))
    joint = t["reaction_conditional"].set_index("given_reaction")
    fig = go.Figure(go.Heatmap(x=joint.columns, y=joint.index, z=joint.to_numpy(),
                               colorbar=dict(title="P(B|A)"), zmin=0, zmax=1,
                               hovertemplate="Дано A=%{y}<br>Реакция B=%{x}<br>P(B|A)=%{z:.3f}<extra></extra>"))
    add("04_reactions", "reaction_conditional", finish(fig,
         "Сочетания реакций: условная частота P(B | A)", "Реакция B", "Дано: реакция A"))
    fig = go.Figure()
    for col in ("like_rate", "dislike_rate", "share_rate", "bookmark_rate"):
        fig.add_trace(go.Scatter(x=ratio.ratio_bin, y=100*ratio[col], mode="lines+markers",
                                 name=col.replace("_rate", ""), customdata=ratio.n,
                                 hovertemplate="Интервал: %{x}<br>Частота: %{y:.2f}%<br>N=%{customdata:,}<extra></extra>"))
    add("04_reactions", "reaction_vs_ratio", finish(fig,
                "Реакции по интервалам доли просмотра (не причинная связь)",
                "Timespent / duration", "Доля событий с реакцией, %"))

    us = t["user_summary"]
    sub = us[us.n_interactions >= 20]
    fig = go.Figure(go.Histogram(x=100*sub.like_rate, nbinsx=40))
    add("05_users", "user_like_rate", finish(fig,
        "Доли лайков у пользователей с ≥20 взаимодействиями (только визуальный фильтр)",
        "Доля лайков, %", "Число пользователей"))
    fig = go.Figure(go.Scattergl(x=us.n_interactions, y=100*us.like_rate, mode="markers",
                                 marker=dict(size=5, opacity=.45)))
    add("05_users", "user_activity_vs_like", finish(fig, "Активность пользователей vs доля лайков",
                                                     "Взаимодействия", "Доля лайков, %"))
    fig = go.Figure(go.Histogram(x=us.mean_ratio.clip(upper=3), nbinsx=60))
    add("05_users", "mean_ratio_users", finish(fig, "Средняя доля просмотра по пользователям (отображение 0–3)",
                                               "Средняя доля просмотра, значения >3 отображены как 3", "Пользователи"))
    fig = go.Figure(go.Scattergl(x=us.n_interactions, y=us.n_authors, mode="markers",
                                 marker=dict(size=5, opacity=.45)))
    add("05_users", "authors_diversity", finish(fig, "Активность vs число уникальных авторов",
                                                 "Взаимодействия", "Авторы"))

    ages = t["age"].sort_values("age")
    add("06_metadata", "age", bars(ages.age, ages.n_users,
                                    "Возраст пользователей (по уникальным пользователям)",
                                    "Возраст, лет", "Пользователи"))
    genders = t["gender"].copy()
    add("06_metadata", "gender", bars(genders.gender.astype(str), genders.n_users,
                                       "Пол (анонимизированные числовые коды)", "Код gender", "Пользователи"))
    geo = t["geo"].nlargest(15, "n_users")
    add("06_metadata", "geo", bars(geo.geo.astype(str), geo.n_users,
                                    "15 крупнейших групп geo", "Код geo", "Пользователи"))
    a = t["authors"]
    fig = go.Figure(go.Histogram(x=a.n_videos, nbinsx=60))
    fig.update_yaxes(type="log")
    add("06_metadata", "author_counts", finish(fig, "Число видео на автора (логарифмическая ось Y)",
                                                 "Видео на автора", "Количество авторов (log)"))
    contexts = t["contexts"]
    fig = go.Figure()
    for variable in ("place", "platform", "agent"):
        subset = contexts[contexts.variable == variable].nlargest(12, "n")
        fig.add_trace(go.Bar(x=[f"{variable}:{v}" for v in subset.code],
                             y=subset.n, name=variable))
    add("06_metadata", "contexts", finish(fig, "Популярные коды контекста показа",
                                            "Переменная:код", "Число взаимодействий"))
    fig = go.Figure()
    for variable in ("place", "platform", "agent"):
        subset = contexts[(contexts.variable == variable) & (contexts.n >= 1000)]
        fig.add_trace(go.Scatter(x=[f"{variable}:{v}" for v in subset.code],
                                 y=100*subset.like_rate, mode="markers", name=variable,
                                 marker=dict(size=9)))
    add("06_metadata", "contexts_like_rate", finish(fig,
        "Доли лайков по кодам контекста (группы N≥1000)",
        "Код контекста", "Доля лайков, %"))

    w = t["weekly"].sort_values("week")
    fig = go.Figure(go.Scatter(x=w.week, y=w.n_events, mode="lines+markers"))
    add("07_time", "weekly_events", finish(fig, "Число взаимодействий по неделям",
                                            "Историческая неделя", "Показы"))
    fig = go.Figure()
    for col, text in [("active_users", "Активные пользователи"), ("active_items", "Активные видео")]:
        fig.add_trace(go.Scatter(x=w.week, y=w[col], mode="lines+markers", name=text))
    add("07_time", "weekly_entities", finish(fig, "Активные пользователи и видео по неделям",
                                              "Неделя", "Уникальные сущности"))
    fig = go.Figure()
    for col, label in [("like_rate", "Лайки"), ("dislike_rate", "Дизлайки"),
                       ("share_rate", "Репосты")]:
        fig.add_trace(go.Scatter(x=w.week, y=100*w[col], mode="lines+markers", name=label))
    add("07_time", "weekly_reactions", finish(fig, "Частота реакций по неделям",
                                               "Неделя", "Доля взаимодействий, %"))
    fig = go.Figure(go.Scatter(x=w.week, y=w.median_watch_ratio, mode="lines+markers"))
    add("07_time", "weekly_ratio", finish(fig, "Медианная доля просмотра по неделям",
                                           "Неделя", "Timespent / duration, медиана"))
    return f
