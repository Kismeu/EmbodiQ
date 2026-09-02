"""Stop VLA-Adapter from rewriting its own checkpoint directory on every load.

``VLA-Adapter/experiments/robot/openvla_utils.py:302-303`` calls, inside
``get_vla()``:

    update_auto_map(cfg.pretrained_checkpoint)
    check_model_logic_mismatch(cfg.pretrained_checkpoint)

which rewrite ``config.json`` and copy the repo's ``modeling_prismatic.py`` into
the checkpoint directory -- every single load. With four concurrent workers
pointed at one checkpoint that is a write race on a 3 GB directory.

We do the sync ONCE offline, hash the directory, freeze it read-only, and then
neutralize both functions at runtime. Nothing in the repo is edited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SKIP = {".git", "__pycache__"}


def _noop(*_args, **_kwargs) -> None:
    return None


def apply(module=None) -> list[str]:
    """Replace the two mutating functions with no-ops. Returns what was patched."""
    if module is None:
        from experiments.robot import openvla_utils as module  # type: ignore

    patched = []
    for name in ("update_auto_map", "check_model_logic_mismatch"):
        if hasattr(module, name):
            setattr(module, name, _noop)
            patched.append(name)
    return patched


def checkpoint_digest(ckpt_dir: str | Path) -> dict:
    """sha256 over (relative path, size) plus content hashes of the small files.

    Large tensors are covered by size alone -- rewriting them is not the failure
    mode we are guarding against; the .py/.json rewrites are, and those are small.
    """
    root = Path(ckpt_dir)
    entries = []
    h_all = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part in _SKIP for part in p.parts):
            continue
        rel = str(p.relative_to(root))
        size = p.stat().st_size
        if size <= 4 << 20:  # hash content of small files (config.json, *.py)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            digest = f"size:{size}"
        entries.append({"path": rel, "size": size, "sha256": digest})
        h_all.update(rel.encode()); h_all.update(digest.encode())
    return {"root": str(root), "n_files": len(entries), "digest": h_all.hexdigest(), "files": entries}


def write_manifest(ckpt_dir: str | Path) -> Path:
    d = checkpoint_digest(ckpt_dir)
    out = Path(ckpt_dir).parent / f".{Path(ckpt_dir).name}.manifest.json"
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return out


def verify_manifest(ckpt_dir: str | Path) -> None:
    """Raise if the directory drifted since freezing. Called from Policy.load()."""
    path = Path(ckpt_dir).parent / f".{Path(ckpt_dir).name}.manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no frozen manifest for {ckpt_dir}; run write_manifest first")
    expect = json.loads(path.read_text(encoding="utf-8"))
    actual = checkpoint_digest(ckpt_dir)
    if actual["digest"] != expect["digest"]:
        old = {f["path"]: f["sha256"] for f in expect["files"]}
        new = {f["path"]: f["sha256"] for f in actual["files"]}
        changed = sorted(
            set(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        )
        raise RuntimeError(
            f"checkpoint {ckpt_dir} changed since freeze; differing entries: {changed[:10]}"
        )
