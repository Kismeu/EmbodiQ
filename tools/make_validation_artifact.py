"""Produce a CONTRACT-VALID artifact for validating the AWQ/HSA pipeline.

This is a plumbing test fixture, not a quantization method. It exists because the
unified entry's AWQ/HSA path cannot be trusted until something has actually gone
through it end to end -- artifact load, hash check, scope conformance, packed
int4 swap with input scales and residuals, fused kernel, rollout, result JSON --
and 1.2/1.3 have not delivered artifacts yet.

WHAT THIS IS NOT:

* Not AWQ. No activations are collected and no reconstruction objective is
  searched. The input scale is a function of the weight magnitudes alone.
* Not HSA. Channels are selected by weight magnitude, which is HSASpec's
  `criterion="magnitude"` ablation arm, with no activation sensitivity and no
  layer-level sensitivity ranking at all.

Every artifact it writes carries ``validation_only: True`` and an
``implementation`` name that says so, and the runs made from it belong in
``results/validation_*`` -- never in the stage directories ``report_matrix.py``
reads. A number produced from this artifact is a statement about the pipeline,
never about a quantization method.

What it does compute correctly, because that is the point:

* the scale is strictly positive, finite, per input channel, normalized the way
  ``awq_blocks.search_block_scale`` normalizes its candidates;
* the residual is in the SCALED weight domain, which is where the fused kernel
  adds it (``dequantized()`` does ``w += res_val`` and only then applies
  ``s_inv``), so the selected channels come back exact and the HSA execution
  path is genuinely exercised rather than merely populated.

Usage:
    python tools/make_validation_artifact.py --model vla_adapter \\
        --suite libero_spatial --method hsa --gpu 3 \\
        --out artifacts/validation/hsa_vla_adapter_spatial.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_unified_baselines as base  # noqa: E402

RECIPE = {
    "awq": "input scale = normalized per-input-channel mean|W| ** alpha",
    "hsa": "awq-shaped scale + exact residual on the top-k channels by mean|W|",
}
ALIGN = 8


def policy_roots(policy, model: str) -> dict:
    """The same roots dict the policy would hand to quant_apply."""
    if model == "vla_adapter":
        roots = {"model": policy.model}
        if policy.action_head is not None:
            roots["action_head"] = policy.action_head
        if policy.proprio_projector is not None:
            roots["proprio_projector"] = policy.proprio_projector
        return roots
    return {"model": policy.model}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=base.REPO.keys())
    ap.add_argument("--suite", required=True, choices=base.SUITES)
    ap.add_argument("--method", required=True, choices=("awq", "hsa"))
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; pick a fresh path")

    import os
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true",
                      TOKENIZERS_PARALLELISM="false")
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()

    import torch

    repo = base.EMB / base.REPO[args.model]
    sys.path.insert(0, str(repo.resolve()))
    from embodiq.entry.run_adapter import MODEL_SCOPES
    from embodiq.quant.artifact import FORMAT, tensor_sha256
    from embodiq.quant.pack import dequantize_int4, group_quantize_int4
    from embodiq.quant.spec import HSASpec, QuantSpec
    from embodiq.quant.walker import plan_quantization

    if args.model == "openvla":
        from embodiq.backends.openvla.policy import OpenVLAPolicy as PolicyClass
        kwargs = {}
    elif args.model == "fastwam":
        from embodiq.backends.fastwam.policy import FastWAMPolicy as PolicyClass
        kwargs = dict(repo_root=repo, seed=42,
                      dataset_stats=base.EMB / "FastWAM/checkpoints/fastwam_release"
                                               "/libero_uncond_2cam224_dataset_stats.json")
    else:
        from embodiq.backends.adapter.policy import VLAAdapterPolicy as PolicyClass
        kwargs = {}

    checkpoint = base.checkpoint(args.model, args.suite)
    # No quant_apply: we need the ORIGINAL Linear weights to hash and to
    # quantize offline, exactly as the evaluator will see them.
    policy = PolicyClass(str(checkpoint), args.suite, device_index=0,
                         quant_apply=None, **kwargs)
    policy.load()
    print(f"loaded {args.model}/{args.suite} from {checkpoint}", flush=True)

    try:
        roots = policy_roots(policy, args.model)
        scope = MODEL_SCOPES[args.model]
        hsa = HSASpec(rho=args.rho) if args.method == "hsa" else None
        spec = QuantSpec(method=args.method, hsa=hsa)
        report = plan_quantization(roots, args.model, spec, enabled_scopes=set(scope))
        targets = [r for r in report.layers if r.decision == "int4"]
        print(f"plan: {len(targets)} int4 layers over scope {scope}", flush=True)
        if not targets:
            raise SystemExit("plan selected no int4 layers")

        group = spec.group_size
        layers: dict[str, dict] = {}
        n_residual_units = 0
        with torch.no_grad():
            for i, rec in enumerate(targets):
                root_name, _, local = (rec.name.partition(".") if len(roots) > 1
                                       else ("", "", rec.name))
                root = roots[root_name] if root_name in roots else next(iter(roots.values()))
                module = root.get_submodule(local or rec.name)
                w = module.weight.detach().float()

                # Scale from weight magnitudes, normalized as the real search
                # normalizes its grid candidates.
                mag = w.abs().mean(dim=0).clamp(min=1e-6)
                s = mag.pow(args.alpha)
                s = (s / (s.max() * s.min()).sqrt()).clamp(min=1e-4)

                row = {
                    "weight_sha256": tensor_sha256(module.weight),
                    "bias_sha256": tensor_sha256(module.bias),
                    "input_scale": s.to(torch.float32).cpu(),
                }

                if args.method == "hsa":
                    # Residual lives in the scaled domain: the kernel adds it
                    # before applying s_inv, so these channels come back exact.
                    k = max(ALIGN, (int(args.rho * module.in_features) // ALIGN) * ALIGN)
                    k = min(k, module.in_features)
                    idx = torch.topk(mag, k).indices.sort().values
                    ws = w * s[None]
                    q, sc, z = group_quantize_int4(ws, group)
                    recon = dequantize_int4(q, sc.bfloat16(), z.bfloat16(), group)
                    row["residual_idx"] = idx.to(torch.int32).cpu()
                    row["residual_val"] = (ws - recon)[:, idx].to(torch.float32).cpu()
                    n_residual_units += k

                layers[rec.name] = row
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(targets)} layers", flush=True)

        recipe = RECIPE[args.method]
        payload = {
            "format": FORMAT,
            "method": args.method,
            # Named so no report can quote it as an AWQ or HSA result.
            "implementation": f"pipeline_validation_{args.method}_v0",
            "implementation_version": "0",
            "validation_only": True,
            "not_a_method_result": (
                "Weight-statistics fixture for validating the unified entry's "
                "artifact path. No activation calibration, no scale search, no "
                "sensitivity ranking. Never cite as an AWQ or HSA result."),
            "producer_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "model_family": args.model,
            "suite": args.suite,
            "w_bit": spec.w_bit,
            "group_size": group,
            "source_checkpoint_sha256": hashlib.sha256(
                json.dumps({"checkpoint": str(checkpoint),
                            "fingerprint": sorted(
                                (n, r.n_params) for n, r in
                                ((x.name, x) for x in targets))},
                           sort_keys=True).encode()).hexdigest(),
            "calibration": {
                # A real hash of a real thing: the recipe, since there is no
                # calibration set. The contract requires the field; filling it
                # with a borrowed calibration hash would be a lie.
                "manifest_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
                "n_calibration": 0,
                "n_development": 0,
                "source": f"none -- {recipe}",
            },
            "scope": list(scope),
            "layers": layers,
            "created": time.strftime("%FT%T%z"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.out)
        size = args.out.stat().st_size
        print(f"wrote {args.out} ({size / 2**20:.1f} MiB): {len(layers)} layers, "
              f"{n_residual_units} residual channels", flush=True)
        print(f"sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}")
        return 0
    finally:
        policy.unload()


if __name__ == "__main__":
    raise SystemExit(main())
