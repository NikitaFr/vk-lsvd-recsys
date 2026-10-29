"""Download + reproducible user/item subsampling of the official VK-LSVD slice.

Only historical weeks 00-24 are touched. No validation/test data, no labels,
no train/val/test generation. Requires pyarrow and huggingface_hub.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
REACTIONS = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")
INTERACTION_COLS = ("user_id", "item_id", "place", "platform", "agent", "timespent", *REACTIONS)
USER_META = ("user_id", "age", "gender", "geo", "train_interactions_rank")
ITEM_META = ("item_id", "author_id", "duration", "train_interactions_rank")


def source_paths(root: Path, cfg: dict) -> list[Path]:
    relative = Path("subsamples") / cfg["subsample"] / "train"
    return [root / "data" / "source" / relative / f"week_{i:02d}.parquet"
            for i in range(cfg["history_weeks"])]


def download_files(root: Path, cfg: dict) -> list[Path]:
    from huggingface_hub import hf_hub_download

    source = root / "data" / "source"
    source.mkdir(parents=True, exist_ok=True)
    revision_file = source / "source_revision.json"
    if revision_file.exists():
        record = json.loads(revision_file.read_text(encoding="utf-8"))
        revision = record["revision"]
        if record.get("repo") != cfg["dataset_repo"] or revision != cfg["source_revision"]:
            raise ValueError("Existing data/source revision differs from the selected dataset revision")
    else:
        revision = cfg["source_revision"]
        revision_file.write_text(json.dumps({"repo": cfg["dataset_repo"],
                                             "revision": revision}, indent=2), encoding="utf-8")
    relative = [f"subsamples/{cfg['subsample']}/train/week_{i:02}.parquet"
                for i in range(cfg["history_weeks"])]
    relative += ["metadata/users_metadata.parquet", "metadata/items_metadata.parquet"]
    LOG.info("Downloading %d official files (cached if previously fetched); commit %s",
             len(relative), revision)
    for index, file in enumerate(relative, 1):
        destination = source / file
        if destination.is_file() and destination.stat().st_size > 0:
            LOG.info("[%d/%d] already downloaded: %s", index, len(relative), file)
            continue
        LOG.info("[%d/%d] fetching: %s", index, len(relative), file)
        hf_hub_download(repo_id=cfg["dataset_repo"], repo_type="dataset",
                        filename=file, revision=revision, local_dir=source)
    return source_paths(root, cfg)


def top_counter(counter: Counter, n: int, seed: int) -> np.ndarray:
    """Uniform independent sampling of IDs seen in the historical data.

    Sorting is important: Counter insertion order depends on Parquet batches.
    """
    ids = np.asarray(sorted(counter), dtype=np.uint64)
    if len(ids) < n:
        raise ValueError(f"Only {len(ids)} distinct IDs available; requested {n}.")
    chosen = np.random.default_rng(seed).choice(ids, size=n, replace=False)
    return np.sort(chosen)


def count_source(paths: list[Path], batch_size: int) -> tuple[Counter, Counter, pd.DataFrame]:
    """Stream batches and aggregate source activity without loading 48M rows into RAM."""
    import pyarrow.parquet as pq
    users, items, weekly = Counter(), Counter(), []
    for week, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"Missing source week: {path}")
        n = 0
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        missing = set(INTERACTION_COLS) - names
        if missing:
            raise ValueError(f"Unexpected parquet schema in {path.name}: missing {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["user_id", "item_id"]):
            u = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
            i = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
            uc, un = np.unique(u, return_counts=True)
            ic, inn = np.unique(i, return_counts=True)
            users.update({int(k): int(v) for k, v in zip(uc, un)})
            items.update({int(k): int(v) for k, v in zip(ic, inn)})
            n += batch.num_rows
        weekly.append({"week": week, "source_interactions": n})
        LOG.info("Source week %02d: %s interactions", week, f"{n:,}")
    return users, items, pd.DataFrame(weekly)


def filter_batch(batch, users: np.ndarray, items: np.ndarray):
    """Vectorised set intersection; batch is a PyArrow RecordBatch."""
    import pyarrow as pa
    u = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
    i = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
    mask = np.isin(u, users) & np.isin(i, items)
    if not np.any(mask):
        return None
    return batch.filter(pa.array(mask))


def write_sample(paths: list[Path], users: np.ndarray, items: np.ndarray,
                 target: Path, batch_size: int) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".partial.parquet")
    writer = None
    counts = []
    try:
        for week, source in enumerate(paths):
            n = 0
            for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size):
                filtered = filter_batch(batch, users, items)
                if filtered is None:
                    continue
                table = pa.Table.from_batches([filtered])
                table = table.append_column("week", pa.array(np.full(len(table), week, dtype=np.int16)))
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                writer.write_table(table)
                n += len(table)
            counts.append({"week": week, "sample_interactions": n})
            LOG.info("Selected week %02d: %s interactions", week, f"{n:,}")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No interactions in selected user/item intersection.")
    tmp_path.replace(target)
    return pd.DataFrame(counts)


def extract_metadata(source: Path, name: str, id_col: str,
                     ids: np.ndarray, output: Path, batch_size: int) -> int:
    """Stream the *full* metadata and retain selected ids only; no 20M-row load."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_path = source / "metadata" / name
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    pieces = []
    for batch in pq.ParquetFile(input_path).iter_batches(batch_size=batch_size):
        arr = batch.column(batch.schema.get_field_index(id_col)).to_numpy(zero_copy_only=False)
        filtered = batch.filter(pa.array(np.isin(arr, ids)))
        if filtered.num_rows:
            pieces.append(pa.Table.from_batches([filtered]))
    if not pieces:
        raise ValueError(f"No selected IDs found in {name}")
    small = pa.concat_tables(pieces)
    if len(np.unique(small[id_col].to_numpy())) != len(small):
        raise ValueError(f"Duplicate metadata IDs found in {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(small, output, compression="zstd")
    return small.num_rows


def make_sample(root: Path, cfg: dict, *, download: bool = True,
                overwrite: bool = False) -> dict:
    root = Path(root)
    sample_dir = root / "data" / "work"
    report_dir = root / "results" / "eda"
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = sample_dir / "manifest.json"
    if manifest_file.exists() and not overwrite:
        raise FileExistsError("A sample already exists. Use --overwrite only if you intend to regenerate it.")
    if download:
        paths = download_files(root, cfg)
    else:
        paths = source_paths(root, cfg)
    user_counts, item_counts, source_week = count_source(paths, int(cfg["batch_size"]))
    users = top_counter(user_counts, int(cfg["sample_users"]), int(cfg["random_seed"]))
    items = top_counter(item_counts, int(cfg["sample_items"]), int(cfg["random_seed"]) + 1)
    sampled_week = write_sample(paths, users, items,
                                sample_dir / "interactions.parquet", int(cfg["batch_size"]))
    full_week = source_week.merge(sampled_week, on="week", validate="one_to_one")
    full_week.to_csv(report_dir / "source_vs_sample_weekly.csv", index=False)
    selected_user_ids = set(int(x) for x in users)
    selected_item_ids = set(int(x) for x in items)
    pd.DataFrame({"user_id": list(user_counts.keys()),
                  "n_interactions": list(user_counts.values()),
                  "selected": [int(u in selected_user_ids) for u in user_counts]})\
        .sort_values("user_id").to_csv(report_dir / "source_user_activity.csv", index=False)
    pd.DataFrame({"item_id": list(item_counts.keys()),
                  "n_interactions": list(item_counts.values()),
                  "selected": [int(i in selected_item_ids) for i in item_counts]})\
        .sort_values("item_id").to_csv(report_dir / "source_item_popularity.csv", index=False)
    source_dir = root / "data" / "source"
    n_users_metadata = extract_metadata(source_dir, "users_metadata.parquet", "user_id", users,
                                        sample_dir / "users_metadata.parquet", int(cfg["batch_size"]))
    n_items_metadata = extract_metadata(source_dir, "items_metadata.parquet", "item_id", items,
                                        sample_dir / "items_metadata.parquet", int(cfg["batch_size"]))
    n_rows = int(sampled_week.sample_interactions.sum())
    # Actual distinct counts may be below sampled ID counts. EDA checks exact values.
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": cfg["dataset_repo"], "original_slice": cfg["subsample"],
                "history_weeks": list(range(cfg["history_weeks"])),
                "sample_users_requested": len(users), "sample_items_requested": len(items),
                "seed_users": cfg["random_seed"], "seed_items": cfg["random_seed"] + 1,
                "interactions_rows": n_rows,
                "source_interactions_rows": int(source_week.source_interactions.sum()),
                "users_metadata_rows": n_users_metadata,
                "items_metadata_rows": n_items_metadata,
                "source_revision": json.loads((source_dir / "source_revision.json").read_text())["revision"]
                                   if (source_dir / "source_revision.json").exists() else "local_unpinned",
                "sampling_description": "Independent uniform sampling of user and item IDs observed in historical weeks 00-24; retain all interactions for selected pairs.",
                "status": "TOO_MANY" if n_rows > 2_000_000 else "TOO_FEW" if n_rows < 100_000 else "ROW_COUNT_OK"}
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Result: %s interactions. Status: %s. Verify detailed conditions in EDA.",
             f"{n_rows:,}", manifest["status"])
    return manifest


CONFIG = {'dataset_repo': 'deepvk/VK-LSVD', 'subsample': 'up0.001_ip0.001', 'history_weeks': 25, 'sample_users': 3000, 'sample_items': 2000, 'random_seed': 42, 'batch_size': 250000, 'recommended_max_interactions': 2000000, 'source_revision': '084a18386e149a454dd6dbe0123478e75c6b09ce'}
