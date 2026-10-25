"""Data-quality audit and EDA aggregates. No model training or target engineering."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preparation import REACTIONS, INTERACTION_COLS


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = Path(root) / "data" / "work"
    if not (work / "interactions.parquet").exists():
        raise FileNotFoundError("Working sample missing; first run: python -m scripts.prepare_vk")
    events = pd.read_parquet(work / "interactions.parquet")
    users = pd.read_parquet(work / "users_metadata.parquet")
    items = pd.read_parquet(work / "items_metadata.parquet")
    required = set(INTERACTION_COLS) | {"week"}
    if required - set(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    if users.user_id.duplicated().any() or items.item_id.duplicated().any():
        raise ValueError("User/item metadata keys are not unique; refusing ambiguous joins")
    return events, users, items


def enrich_events(events: pd.DataFrame, users: pd.DataFrame,
                  items: pd.DataFrame) -> pd.DataFrame:
    user_cols = [c for c in ("user_id", "age", "gender", "geo") if c in users]
    item_cols = [c for c in ("item_id", "author_id", "duration") if c in items]
    df = events.merge(users[user_cols], on="user_id", how="left", validate="many_to_one")
    df = df.merge(items[item_cols], on="item_id", how="left", validate="many_to_one")
    valid_duration = df["duration"].notna() & (df["duration"] > 0)
    df["watch_ratio"] = np.nan
    df.loc[valid_duration, "watch_ratio"] = (
        df.loc[valid_duration, "timespent"] / df.loc[valid_duration, "duration"]
    )
    return df


def _native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def rate_by_ratio(df: pd.DataFrame) -> pd.DataFrame:
    edges = np.array([-np.inf, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1,
                      1.25, 1.5, 2, 3, np.inf])
    labels = ["0", "(0, .1]", "(.1, .2]", "(.2, .3]", "(.3, .4]",
              "(.4, .5]", "(.5, .6]", "(.6, .7]", "(.7, .8]",
              "(.8, .9]", "(.9, 1]", "(1, 1.25]", "(1.25, 1.5]",
              "(1.5, 2]", "(2, 3]", ">3"]
    # Distinguish zero values from NaN durations. (-inf, 0] includes only 0 for valid watch ratio.
    bins = pd.cut(df.watch_ratio, bins=edges, labels=labels, include_lowest=True)
    base = pd.DataFrame({"ratio_bin": bins})
    for col in REACTIONS:
        base[col] = pd.to_numeric(df[col], errors="coerce")
    aggregations = {"n": ("like", "size")}
    for col in REACTIONS:
        aggregations[f"{col}_rate"] = (col, "mean")
        aggregations[f"{col}_count"] = (col, "sum")
    result = base.groupby("ratio_bin", observed=False).agg(**aggregations).reset_index()
    result["ratio_bin"] = result["ratio_bin"].astype(str)
    return result


def build_tables(events: pd.DataFrame, users: pd.DataFrame,
                 items: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Produce all numerical aggregates that back the figures and conclusions."""
    df = enrich_events(events, users, items)
    n_events = len(df)
    n_users = int(df.user_id.nunique())
    n_items = int(df.item_id.nunique())
    assert n_events > 0, "No sample interactions"

    missing = df.isna().sum().rename("missing").to_frame()
    missing["fraction"] = missing["missing"] / n_events
    missing.index.name = "column"
    missing = missing.reset_index()

    user_activity = df.groupby("user_id").size().rename("n_interactions").reset_index()
    item_popularity = df.groupby("item_id").size().rename("n_interactions").reset_index()

    user_summary = df.groupby("user_id").agg(
        n_interactions=("item_id", "size"),
        n_authors=("author_id", "nunique"),
        mean_ratio=("watch_ratio", "mean"),
        median_ratio=("watch_ratio", "median"),
        **{f"{c}_rate": (c, "mean") for c in REACTIONS},
    ).reset_index()

    item_summary = df.groupby("item_id").agg(
        n_interactions=("user_id", "size"),
        mean_ratio=("watch_ratio", "mean"),
        **{f"{c}_rate": (c, "mean") for c in REACTIONS},
    ).reset_index()
    item_summary = item_summary.merge(
        items[["item_id", "author_id", "duration"]].drop_duplicates("item_id"),
        on="item_id", how="left", validate="one_to_one")

    reactions = pd.DataFrame({
        "reaction": list(REACTIONS),
        "count": [int(df[c].fillna(False).sum()) for c in REACTIONS],
        "rate": [float(df[c].mean()) for c in REACTIONS],
        "missing": [int(df[c].isna().sum()) for c in REACTIONS],
    })
    binary = np.column_stack([df[c].fillna(False).to_numpy(dtype=np.int64)
                              for c in REACTIONS])
    joint = binary.T @ binary
    conditional = np.divide(joint, joint.diagonal()[:, None],
                            out=np.zeros_like(joint, dtype=float),
                            where=joint.diagonal()[:, None] > 0)
    joint_df = pd.DataFrame(conditional, index=REACTIONS, columns=REACTIONS)
    joint_counts = pd.DataFrame(joint, index=REACTIONS, columns=REACTIONS)

    weekly_agg = {"n_events": ("user_id", "size"),
                  "active_users": ("user_id", "nunique"),
                  "active_items": ("item_id", "nunique"),
                  "median_watch_ratio": ("watch_ratio", "median")}
    weekly_agg.update({f"{c}_rate": (c, "mean") for c in REACTIONS})
    weekly = df.groupby("week").agg(**weekly_agg).reset_index().sort_values("week")

    ts, ts_edges = np.histogram(df.timespent.dropna().to_numpy(), bins=np.arange(0, 261, 5))
    timespent_hist = pd.DataFrame({"left": ts_edges[:-1], "right": ts_edges[1:], "count": ts})
    di, d_edges = np.histogram(items.duration.dropna().to_numpy(), bins=np.arange(0, 261, 5))
    duration_hist = pd.DataFrame({"left": d_edges[:-1], "right": d_edges[1:], "count": di})

    ratio_bins = rate_by_ratio(df)
    duration_bins = pd.cut(df.duration, bins=[-0.01, 15, 30, 60, 120, 180, 255],
                           labels=["0–15", "16–30", "31–60", "61–120", "121–180", "181–255"])
    by_duration = df.assign(duration_group=duration_bins).groupby("duration_group", observed=False).agg(
        n=("item_id", "size"), median_ratio=("watch_ratio", "median"),
        mean_ratio=("watch_ratio", "mean"), like_rate=("like", "mean")
    ).reset_index()
    by_duration["duration_group"] = by_duration.duration_group.astype(str)
    watch_bins = pd.cut(df.watch_ratio, bins=[-.001, .1, .25, .5, .75, 1, 1.25, 2, np.inf],
                        labels=["0–10%", "10–25%", "25–50%", "50–75%", "75–100%",
                                "100–125%", "125–200%", ">200%"])
    watch_duration_counts = pd.crosstab(duration_bins, watch_bins, dropna=False)
    # Normalize rows for a meaningful cross-duration comparison; raw counts are saved too.
    watch_duration_dist = watch_duration_counts.div(
        watch_duration_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    contexts = []
    for name in ("place", "platform", "agent"):
        a = df.groupby(name, dropna=False).agg(n=("user_id", "size"),
                                                like_rate=("like", "mean"),
                                                dislike_rate=("dislike", "mean")).reset_index()
        a["variable"] = name
        a = a.rename(columns={name: "code"})
        contexts.append(a)
    contexts = pd.concat(contexts, ignore_index=True)

    authors = items.groupby("author_id", dropna=False).agg(n_videos=("item_id", "size"),
                                                            median_duration=("duration", "median")).reset_index()
    age = users.groupby("age", dropna=False).size().rename("n_users").reset_index()
    gender = users.groupby("gender", dropna=False).size().rename("n_users").reset_index()
    geography = users.groupby("geo", dropna=False).size().rename("n_users").reset_index()

    # Group users/items into activity-ranked bins, then count observed matrix cells.
    # Each cell represents the observed density among the corresponding entity groups.
    k = 20
    us = user_activity.sort_values(["n_interactions", "user_id"], ascending=[False, True]).copy()
    it = item_popularity.sort_values(["n_interactions", "item_id"], ascending=[False, True]).copy()
    us["user_bin"] = np.minimum(k - 1, np.arange(len(us)) * k // len(us))
    it["item_bin"] = np.minimum(k - 1, np.arange(len(it)) * k // len(it))
    ub = dict(zip(us.user_id, us.user_bin))
    ib = dict(zip(it.item_id, it.item_bin))
    observed = pd.DataFrame({"u": df.user_id.map(ub), "i": df.item_id.map(ib)})
    grid = pd.crosstab(observed.u, observed.i).reindex(index=range(k), columns=range(k), fill_value=0)
    u_sizes = us.groupby("user_bin").size().reindex(range(k), fill_value=0).to_numpy()
    i_sizes = it.groupby("item_bin").size().reindex(range(k), fill_value=0).to_numpy()
    denominator = np.outer(u_sizes, i_sizes)
    heat = np.divide(grid.to_numpy(), denominator, out=np.zeros((k, k)), where=denominator > 0)
    matrix_density = pd.DataFrame(heat, index=[f"U{i+1}" for i in range(k)],
                                  columns=[f"I{i+1}" for i in range(k)])

    checks = {
        "events": n_events,
        "unique_users": n_users,
        "unique_items": n_items,
        "matrix_density": n_events / (n_users * n_items),
        "duplicate_user_item_pairs": int(df.duplicated(["user_id", "item_id"]).sum()),
        "like_and_dislike": int((df.like.fillna(False) & df.dislike.fillna(False)).sum()),
        "timespent_255": int((df.timespent == 255).sum()),
        "timespent_zero": int((df.timespent == 0).sum()),
        "duration_zero_rows": int((df.duration == 0).sum()),
        "missing_user_metadata_rows": int(df.age.isna().sum()),
        "missing_item_metadata_rows": int(df.duration.isna().sum()),
        "watch_ratio_over_1": int((df.watch_ratio > 1).sum()),
        "watch_ratio_available": int(df.watch_ratio.notna().sum()),
        "share_of_ratio_over_1_among_valid": float((df.watch_ratio > 1).sum() / max(df.watch_ratio.notna().sum(), 1)),
        "age_outside_official_18_70": int((df.age.notna() & ~df.age.between(18, 70)).sum()),
        "source_weeks_observed": sorted(int(x) for x in df.week.unique()),
        "assignment_size_ok": bool(100000 <= n_events <= 2000000 and n_users >= 1000 and n_items >= 1000),
    }
    # No implicit cleaning, dropping, or imputation in this stage.
    tables = {
        "missingness": missing, "user_activity": user_activity,
        "item_popularity": item_popularity, "user_summary": user_summary,
        "item_summary": item_summary, "reactions": reactions,
        "reaction_conditional": joint_df.reset_index().rename(columns={"index": "given_reaction"}),
        "reaction_joint_counts": joint_counts.reset_index().rename(columns={"index": "reaction_A"}),
        "weekly": weekly, "timespent_hist": timespent_hist,
        "duration_hist": duration_hist, "watch_ratio_reactions": ratio_bins,
        "by_duration": by_duration,
        "watch_duration_counts": watch_duration_counts.reset_index().rename(columns={"duration": "duration_group"}),
        "watch_duration_distribution": watch_duration_dist.reset_index().rename(columns={"duration": "duration_group"}),
        "contexts": contexts,
        "authors": authors, "age": age, "gender": gender, "geo": geography,
        "matrix_density_grid": matrix_density.reset_index().rename(columns={"index": "user_activity_bin"}),
    }
    return tables, checks


def run_eda(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict]:
    """Read prepared data; export stats/CSVs/interactive charts; return figures for Jupyter."""
    from .plots import create_figures

    root = Path(root)
    results = root / "results" / "eda"
    distribution_dir = results / "distributions"
    figure_dir = results / "figures"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    events, users, items = load_data(root)
    tables, checks = build_tables(events, users, items)
    (results / "eda_summary.json").write_text(json.dumps(_native(checks), indent=2,
                             ensure_ascii=False), encoding="utf-8")
    (results / "data_checks.json").write_text(json.dumps(_native(checks), indent=2,
                             ensure_ascii=False), encoding="utf-8")
    for name, frame in tables.items():
        frame.to_csv(distribution_dir / f"{name}.csv", index=False)
    source = {}
    for name in ("source_user_activity", "source_item_popularity", "source_vs_sample_weekly"):
        path = results / f"{name}.csv"
        if path.exists():
            source[name] = pd.read_csv(path)
    figures = create_figures(tables, checks, source)
    for section_figures in figures.values():
        for name, figure in section_figures:
            figure.write_html(figure_dir / f"{name}.html", include_plotlyjs="cdn", full_html=True)
    return tables, checks, figures
