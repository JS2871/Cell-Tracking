"""Run the vendored DL baseline's training script for one tiny CPU smoke test.

`scripts/train_unet_transformer.py` (vendored from `royerlab/kaggle-cell-tracking-competition`)
calls `torch.cuda.synchronize()` unconditionally at a few points (meant purely as
GPU timing barriers), which raises on a CPU-only torch build. Rather than
patching the vendored file, this shims `torch.cuda.synchronize` to a no-op
in-process (only when no CUDA device is present) before running the script via
`runpy`, so the real training/detection/edge-prediction code path executes
unmodified on CPU.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_smoke_test(
    repo_root: Path | str,
    debug_video: Path | str,
    epochs: int = 1,
    max_iters: int = 1,
) -> None:
    """Run one tiny debug-video training pass of the vendored DL baseline.

    Parameters
    ----------
    repo_root : Path or str
        Path to the vendored `kaggle-cell-tracking-competition` checkout.
    debug_video : Path or str
        Path to a single dataset (with or without extension) to train+eval on.
    epochs, max_iters : int
        Kept at 1 for a fast sanity check, not a real training run.
    """
    repo_root = Path(repo_root)
    scripts_dir = repo_root / "scripts"
    for p in (str(scripts_dir), str(repo_root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch

    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *a, **kw: None

    sys.argv = [
        "train_unet_transformer.py",
        "--debug-video", str(debug_video),
        "--epochs", str(epochs),
        "--max-iters", str(max_iters),
        "--num-workers", "0",
        "--batch-size", "1",
    ]
    runpy.run_path(str(scripts_dir / "train_unet_transformer.py"), run_name="__main__")
