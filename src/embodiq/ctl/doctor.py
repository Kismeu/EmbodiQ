"""V0: per-environment preflight. Runs INSIDE a model prefix.

Every check here corresponds to a failure mode that produces a plausible-looking
WRONG NUMBER rather than a crash, which is why they are gates and not hygiene.

    python -m embodiq.ctl.doctor --family adapter
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Derived from this file rather than hardcoded: a literal here is what broke
# every tool when the project directory was renamed. The env override stays.
EMB_ROOT = Path(os.environ.get("EMB_ROOT") or Path(__file__).resolve().parents[4])

EXPECTED = {
    "openvla": {
        "repo": "openvla",
        "torch": "2.2", "transformers": "4.40.1", "numpy": "1.26",
        "timm": {"0.9.10", "0.9.11", "0.9.12", "0.9.16"},
    },
    "adapter": {
        "repo": "VLA-Adapter",
        "torch": "2.2", "transformers": "4.40.1", "numpy": "1.26",
        "timm": {"0.9.10", "0.9.11", "0.9.12", "0.9.16"},
    },
    "fastwam": {
        "repo": "FastWAM",
        "torch": "2.7", "transformers": "4.49.0", "numpy": "2.2",
        "timm": None,
    },
}


class Doctor:
    def __init__(self, family: str) -> None:
        self.family = family
        self.exp = EXPECTED[family]
        self.results: list[dict] = []

    def check(self, name: str, fn) -> None:
        try:
            detail = fn()
            self.results.append({"check": name, "ok": True, "detail": detail})
        except Exception as exc:  # noqa: BLE001
            self.results.append({"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    # -- individual checks ------------------------------------------------
    def _versions(self) -> str:
        import numpy, torch

        out = [f"torch={torch.__version__}", f"numpy={numpy.__version__}"]
        assert torch.__version__.startswith(self.exp["torch"]), f"torch {torch.__version__}"
        assert numpy.__version__.startswith(self.exp["numpy"]), f"numpy {numpy.__version__}"
        import transformers

        out.append(f"transformers={transformers.__version__}")
        assert transformers.__version__ == self.exp["transformers"], transformers.__version__
        if self.exp["timm"]:
            import timm

            out.append(f"timm={timm.__version__}")
            # Hard gate: modeling_prismatic raises NotImplementedError off-whitelist.
            assert timm.__version__ in self.exp["timm"], f"timm {timm.__version__} will hard-fail"
        return " ".join(out)

    def _tinygemm(self) -> str:
        """The whole packed-INT4 plan rests on this op existing in BOTH stacks.

        Exercised through embodiq.quant.pack, not hand-rolled: the two torch
        versions take different input formats (int32 [N,K] on 2.2, uint8 [N,K/2]
        on >=2.5) and the packer auto-detects. Hand-rolling here would test the
        wrong thing.
        """
        import torch

        for op in ("_weight_int4pack_mm", "_convert_weight_to_int4pack"):
            assert hasattr(torch.ops.aten, op), f"missing torch.ops.aten.{op}"
        if not torch.cuda.is_available():
            return "ops present (no CUDA here to exercise them)"
        from embodiq.quant.pack import _detect_pack_format, verify_roundtrip

        torch.manual_seed(0)
        w = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.02
        r = verify_roundtrip(w)
        assert r["kernel_matches"], f"packed kernel disagrees with dequant: {r}"
        assert 0.005 < r["weight_rel_fro"] < 0.35, r["weight_rel_fro"]
        return (f"format={_detect_pack_format()} kernel_rel={r['kernel_vs_dequant_rel']:.1e} "
                f"w_rel_fro={r['weight_rel_fro']:.4f}")

    def _import_isolation(self) -> str:
        """prismatic must resolve to THIS prefix's repo, not the sibling one.

        openvla and VLA-Adapter both editable-install a top-level `prismatic`,
        so a shared prefix silently shadows one with the other.
        """
        if self.family == "fastwam":
            mod = importlib.import_module("fastwam")
            path = Path(mod.__file__).resolve()
            assert "FastWAM" in path.parts, f"fastwam resolved to {path}"
            return str(path)
        mod = importlib.import_module("prismatic")
        path = Path(mod.__file__ or mod.__path__[0]).resolve()
        want = self.exp["repo"]
        assert want in path.parts, f"prismatic resolved to {path}, expected inside {want}/"
        return str(path)

    def _tf_off_gpu(self) -> str:
        if self.family == "fastwam":
            return "n/a (no tensorflow in this stack)"
        from embodiq.backends.compat import tf_cpu_guard

        tf_cpu_guard.apply()
        tf_cpu_guard.assert_tf_off_gpu()
        import tensorflow as tf

        return f"tf={tf.__version__}, visible GPUs=0"

    def _egl(self) -> str:
        """Actually render a frame. EGL failures are the classic headless trap."""
        assert os.environ.get("MUJOCO_GL") == "egl", "MUJOCO_GL is not egl"
        from libero.libero import benchmark, get_libero_path  # noqa: F401
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        try:
            env.seed(0)
            obs = env.reset()
            img = obs["agentview_image"]
            assert img.shape == (256, 256, 3), img.shape
            assert img.max() > 0, "rendered frame is all black -- EGL is not really working"
        finally:
            env.close()
        return f"rendered {img.shape}, max={int(img.max())}"

    def _nvml(self) -> str:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        return f"{n} device(s)"

    def _repos_clean(self) -> str:
        """The zero-upstream-edit invariant, checked mechanically.

        Note `pip install -e` regenerates <pkg>.egg-info/, which openvla and
        VLA-Adapter unusually keep tracked. env/protect_upstream.sh marks those
        paths skip-worktree so an editable install cannot dirty the repo; if
        this check fires on an egg-info path, that script has not been run.
        """
        dirty = {}
        for repo in ("openvla", "VLA-Adapter", "FastWAM"):
            p = EMB_ROOT / repo
            if not p.exists():
                continue
            out = subprocess.run(
                ["git", "-C", str(p), "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if out:
                dirty[repo] = out.splitlines()[:5]
        assert not dirty, f"upstream repos modified: {dirty}"
        return "openvla / VLA-Adapter / FastWAM all clean"

    def run(self) -> int:
        self.check("versions", self._versions)
        self.check("tinygemm_int4", self._tinygemm)
        self.check("import_isolation", self._import_isolation)
        self.check("tf_off_gpu", self._tf_off_gpu)
        self.check("egl_render", self._egl)
        self.check("nvml", self._nvml)
        self.check("upstream_repos_clean", self._repos_clean)

        width = max(len(r["check"]) for r in self.results)
        for r in self.results:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['check']:<{width}}  {r['detail']}")
        n_fail = sum(1 for r in self.results if not r["ok"])
        print(f"\n{len(self.results) - n_fail}/{len(self.results)} checks passed for '{self.family}'")
        return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="embodiq environment preflight")
    ap.add_argument("--family", required=True, choices=sorted(EXPECTED))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    d = Doctor(args.family)
    code = d.run()
    if args.json:
        args.json.write_text(json.dumps({"family": args.family, "results": d.results}, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
