"""Download the pinned VK-LSVD subset and build the working sample."""
from pathlib import Path
import json
from .utils.preparation import CONFIG, make_sample


def prepare(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    cfg = CONFIG.copy()
    work = root / 'data/work'
    names = ('interactions.parquet', 'users_metadata.parquet', 'items_metadata.parquet', 'manifest.json')
    present = [(work / name).is_file() for name in names]
    if any(present) and not all(present):
        raise FileNotFoundError('Incomplete data/work; restore the complete working sample')
    if all(present):
        summary = json.loads((work / 'manifest.json').read_text())
        if summary.get('source_revision') != cfg['source_revision'] or summary.get('source') != cfg['dataset_repo']:
            raise ValueError('Existing sample uses a different dataset revision')
    else:
        summary = make_sample(root, cfg)
    # Reproduction scripts record the sample parameters alongside generated data.
    work = root / 'data/work'
    work.mkdir(parents=True, exist_ok=True)
    (work / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    from huggingface_hub import hf_hub_download
    for split, week in [('validation', 25), ('test', 26)]:
        hf_hub_download(repo_id=cfg['dataset_repo'], repo_type='dataset',
                        filename=f"subsamples/{cfg['subsample']}/{split}/week_{week}.parquet",
                        revision=cfg['source_revision'], local_dir=root/'data/source')
    return summary


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
