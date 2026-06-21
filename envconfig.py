"""Machine-dependent paths via a local, git-ignored ``env.yaml``.

This repo runs on several servers (dragon, tiger, ...) where the dataset lives in
different places, while all training outputs are collected on one shared NFS dir.
Those machine-specific paths live in ``env.yaml`` (NOT committed) instead of being
hardcoded. Copy ``env.yaml.example`` to ``env.yaml`` and edit it per machine.

Keys (all required):
  - ``data_root``      : the ghost_dataset directory on THIS machine
  - ``output_root``    : where checkpoints/logs are written (shared NFS, same on all machines)
  - ``cache_root``     : local fast-disk scratch for regenerable representation caches
  - ``topm_repo_root`` : the read-only Ghost-FWL / FWL-ToPM repo (PYTHONPATH + ToPM weights)

Values are passed through ``os.path.expandvars`` and ``os.path.expanduser`` so
``$HOME``/``${VAR}``/``~`` work. Use the helpers (``data_path`` etc.) to build paths.
"""
from __future__ import annotations

import functools
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ENV_YAML = os.path.join(_HERE, "env.yaml")
ENV_EXAMPLE = os.path.join(_HERE, "env.yaml.example")
REQUIRED_KEYS = ("data_root", "output_root", "cache_root", "topm_repo_root")


@functools.lru_cache(maxsize=1)
def load_env() -> dict:
    """Load and validate env.yaml (cached). Raises a friendly error if missing/incomplete."""
    if not os.path.exists(ENV_YAML):
        raise FileNotFoundError(
            f"\n\nMachine-path config '{ENV_YAML}' not found.\n"
            f"It holds per-machine paths and is intentionally git-ignored.\n"
            f"Create it for THIS machine by copying the template and editing it:\n\n"
            f"    cp {ENV_EXAMPLE} {ENV_YAML}\n\n"
            f"then set data_root / output_root / cache_root / topm_repo_root.\n"
        )
    import yaml
    with open(ENV_YAML) as f:
        env = yaml.safe_load(f) or {}
    missing = [k for k in REQUIRED_KEYS if not env.get(k)]
    if missing:
        raise KeyError(
            f"{ENV_YAML} is missing required key(s): {missing}. "
            f"See {ENV_EXAMPLE} for all four keys and their meaning."
        )
    return {k: os.path.expanduser(os.path.expandvars(str(env[k]))) for k in REQUIRED_KEYS}


def data_root() -> str:
    return load_env()["data_root"]


def output_root() -> str:
    return load_env()["output_root"]


def cache_root() -> str:
    return load_env()["cache_root"]


def topm_repo_root() -> str:
    return load_env()["topm_repo_root"]


def data_path(*parts: str) -> str:
    return os.path.join(data_root(), *parts)


def output_path(*parts: str) -> str:
    return os.path.join(output_root(), *parts)


def cache_path(*parts: str) -> str:
    return os.path.join(cache_root(), *parts)


def topm_path(*parts: str) -> str:
    return os.path.join(topm_repo_root(), *parts)


def remap_data_dir(path: str) -> str:
    """Rewrite an absolute dataset path (``…/ghost_dataset/<rel>``) to THIS machine's
    ``data_root``, and apply the annotation_v1 -> annotation_v1_expand fix (the test
    metric uses the *expand* annotations). Behaviour-preserving: if ``data_root`` already
    equals the path's ``…/ghost_dataset`` prefix, the path is unchanged.

    A path without a ``ghost_dataset/`` segment is returned with only the annotation fix.
    """
    if "ghost_dataset/" in path:
        rel = path.split("ghost_dataset/", 1)[1]
        path = os.path.join(data_root(), rel)
    return path.replace("/annotation_v1/", "/annotation_v1_expand/")


def remap_topm(path: str) -> str:
    """Rebase a path inside the FWL-ToPM repo (e.g. a config's ``checkpoint_path`` like
    ``…/<repo>/checkpoints/…pth``) onto THIS machine's ``topm_repo_root``. Keyed on the
    ``/checkpoints/`` marker so it is independent of the repo directory name."""
    i = path.find("/checkpoints/")
    return topm_path(path[i + 1:]) if i >= 0 else path
