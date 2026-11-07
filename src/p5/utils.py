from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger("p5")


def setup_logging(log_file: Path | None = None) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    root = logging.getLogger("p5")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    for k, v in (overrides or {}).items():
        _set_nested(cfg, k, v)
    root = path.resolve().parent
    for key in ("data_dir", "output_dir", "checkpoint_dir", "cache_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else (root / p))
    return cfg


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    import torch

    if prefer != "cpu" and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        free, total = torch.cuda.mem_get_info(dev)
        LOG.info(
            "GPU: %s | VRAM total %.1f GB, free %.1f GB | CUDA %s | bf16 %s",
            props.name, total / 2**30, free / 2**30, torch.version.cuda,
            torch.cuda.is_bf16_supported(),
        )
        return dev
    if prefer == "cuda":
        LOG.warning("CUDA requested but not available - falling back to CPU")
    LOG.warning("Running on CPU. BERT extraction and probe training will be slow.")
    return torch.device("cpu")


def gpu_mem_gb(device) -> float:
    import torch

    if device.type != "cuda":
        return 0.0
    return torch.cuda.mem_get_info(device)[1] / 2**30


def amp_dtype(device):
    import torch

    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Expected file is missing: {path}. Run the earlier task first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def ensure_dirs(cfg: dict) -> dict[str, Path]:
    p = {k: Path(v) for k, v in cfg["paths"].items()}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p
