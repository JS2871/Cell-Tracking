"""Fixed, stratified train/validation split over the 199 train samples.

Stratifies by the two dataset-source prefixes found during EDA (``44b6``
dense, ``6bba`` sparser) so the held-out validation slice has proportional
representation of both. Written in the exact ``dataset_splits.json`` schema
``scripts/train_unet_transformer.py --splits`` expects
(``[{"split": 0, "train": [...], "test": [...]}]``), so a future DL training
run can reuse the identical held-out set.
"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path


def list_train_sample_ids(zip_path: Path | str) -> list[str]:
    """Return sorted train sample ids (basenames, no extension) from the zip."""
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    bases = {n.split("/")[1] for n in names if n.startswith("train/") and "/" in n[len("train/"):]}
    return sorted({b.rsplit(".", 1)[0] for b in bases})


def make_split(
    zip_path: Path | str,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict:
    """Build a stratified-by-prefix train/val split over all train samples.

    Returns
    -------
    dict
        ``{"train": [...ids], "val": [...ids]}``.
    """
    ids = list_train_sample_ids(zip_path)
    by_prefix: dict[str, list[str]] = {}
    for sid in ids:
        by_prefix.setdefault(sid.split("_")[0], []).append(sid)

    train_ids: list[str] = []
    val_ids: list[str] = []
    rng = random.Random(seed)
    for prefix, group in sorted(by_prefix.items()):
        group = sorted(group)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val_ids.extend(group[:n_val])
        train_ids.extend(group[n_val:])

    return {"train": sorted(train_ids), "val": sorted(val_ids)}


def write_dataset_splits_json(split: dict, output_path: Path | str) -> Path:
    """Write `split` in the `train_unet_transformer.py --splits` schema."""
    output_path = Path(output_path)
    folds = [{"split": 0, "train": split["train"], "test": split["val"]}]
    output_path.write_text(json.dumps(folds, indent=2))
    return output_path
