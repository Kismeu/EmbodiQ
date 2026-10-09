"""Load and validate managed experiment configurations.

The evaluator still accepts its existing CLI for compatibility.  This module
provides the stable configuration contract used by future ``run`` orchestration:
model adapters, quantizers and benchmarks are declared data, while execution
remains in the existing entry points.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "embodiq/experiment-config/1"
MODELS = {"openvla", "vla_adapter", "fastwam"}
METHODS = {"fp16", "rtn", "awq", "hsa"}
SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10"}
_SCOPE_KEYS = {"openvla": {"llm"}, "vla_adapter": {"llm", "vision_dino", "vision_siglip", "projector", "action_head"}, "fastwam": {"video_dit", "action_dit"}}
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    defaults = {
        "EMB_ROOT": "/data/jyc/space/Embodied AI",
        "DATA_ROOT": "/data/jyc/data",
        "OPENVLA_ROOT": "/data/jyc/models",
    }
    if isinstance(value, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), defaults.get(m.group(1), m.group(0))), value)
    if isinstance(value, list):
        return [_expand(x) for x in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configs require PyYAML; use JSON otherwise") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config suffix: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("experiment config must be a mapping")
    return _expand(data)


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    protocol = config.get("protocol")
    if protocol != "unified.v3":
        errors.append("protocol must be 'unified.v3' for the formal matrix")
    suites = config.get("benchmark", {}).get("suites", [])
    if set(suites) - SUITES or not suites:
        errors.append(f"benchmark.suites must be drawn from {sorted(SUITES)}")
    profile = config.get("profile", "formal")
    if profile not in {"formal", "smoke"}:
        errors.append("profile must be formal or smoke")
    if profile == "formal" and config.get("trials_per_task") != 50:
        errors.append("trials_per_task must be 50 for the formal matrix")
    if profile == "smoke" and config.get("trials_per_task") != 1:
        errors.append("smoke profile requires trials_per_task=1")
    if config.get("seed") != 42:
        errors.append("seed must be 42 for the frozen episode manifest")
    if config.get("benchmark", {}).get("name") != "libero":
        errors.append("only the libero benchmark is supported")
    if len(suites) != len(set(suites)):
        errors.append("benchmark.suites must not contain duplicates")
    models = config.get("models", {})
    if not models or set(models) - MODELS:
        errors.append(f"models must be a nonempty subset of {sorted(MODELS)}")
    for family in set(models) & MODELS:
        row = models.get(family)
        if not isinstance(row, dict):
            errors.append(f"models.{family} is required")
            continue
        scopes = set(row.get("scopes", []))
        if scopes != _SCOPE_KEYS[family]:
            errors.append(f"models.{family}.scopes must equal {sorted(_SCOPE_KEYS[family])}")
    methods = config.get("enabled_methods", [])
    if not methods or set(methods) - METHODS:
        errors.append(f"enabled_methods must be drawn from {sorted(METHODS)}")
    if len(methods) != len(set(methods)):
        errors.append("enabled_methods must not contain duplicates")
    quant = config.get("quantization", {})
    for method in methods:
        row = quant.get(method)
        if not isinstance(row, dict) or row.get("method") != method:
            errors.append(f"quantization.{method} must declare method={method!r}")
    hsa = quant.get("hsa", {})
    if not hsa.get("enabled", False) and "hsa" in methods:
        # HSA can be listed as a reserved row, but must never be scheduled while
        # its implementation is pending.
        if hsa.get("status") != "pending_external_implementation":
            errors.append("disabled HSA must declare pending_external_implementation")
    for method in ("awq", "hsa"):
        q = quant.get(method, {})
        if method in methods and q.get("enabled", False) and not (q.get("artifact_root") or q.get("artifact")):
            errors.append(f"quantization.{method}.artifact_root or artifact is required")
    if hsa.get("enabled", False):
        rho = hsa.get("rho", 0.01)
        if not isinstance(rho, (int, float)) or not 0 < rho <= 0.08:
            errors.append("quantization.hsa.rho must be in (0, 0.08]")
    return errors


def validate_file(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    errors = validate_config(config)
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return config


def build_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a validated config into executable model/method/suite jobs.

    Disabled methods are retained in the configuration for traceability but are
    never scheduled.  This is especially important for the current HSA pending
    row: it remains visible as a reserved method without producing fake jobs.
    """
    errors = validate_config(config)
    if errors:
        raise ValueError("invalid experiment config: " + "; ".join(errors))
    jobs: list[dict[str, Any]] = []
    suites = config["benchmark"]["suites"]
    for family, model in config["models"].items():
        for method in config["enabled_methods"]:
            q = config["quantization"][method]
            if not q.get("enabled", True):
                continue
            if method == "hsa" and q.get("status") == "pending_external_implementation":
                continue
            for suite in suites:
                job = {
                    "model": family,
                    "suite": suite,
                    "quant": method,
                    "protocol": config["protocol"],
                    "seed": config["seed"],
                    "trials": config["trials_per_task"],
                    "checkpoint": model.get("checkpoint"),
                    "checkpoint_root": model.get("checkpoint_root"),
                    "dataset_stats": model.get("dataset_stats"),
                    "repo_root": model.get("repo_root"),
                    "scopes": model["scopes"],
                    "artifact_root": q.get("artifact_root"),
                    "artifact": q.get("artifact"),
                    "hsa_rho": q.get("rho", 0.01) if method == "hsa" else None,
                    "fastwam_mode": config["runtime"].get("fastwam_mode", "first_frame"),
                    "task_ids": [0] if config.get("profile") == "smoke" else list(range(10)),
                    "checkpoint_every": config.get("checkpoint_every", 1),
                    "save_videos": config.get("save_videos", False),
                }
                jobs.append(job)
    return jobs
