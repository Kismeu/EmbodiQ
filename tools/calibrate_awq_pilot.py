"""Pilot AWQ scale search on real training activations; no deployed AWQ claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise RuntimeError("Use a fresh pilot artifact directory")
    import numpy as np
    import torch
    from torch.nn import functional as F
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()
    sys.path.insert(0, str(ROOT.parent / "VLA-Adapter"))
    from embodiq.backends.adapter.policy import VLAAdapterPolicy
    from embodiq.core.rollout import build_observation
    from embodiq.quant.calibration import ActivationCollector
    from embodiq.quant.awq_search import search_input_scale, IMPLEMENTATION
    from embodiq.quant.pack import group_quantize_int4, dequantize_int4
    from experiments.robot.robot_utils import set_seed_everywhere
    manifest_path = args.observations / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"] != "official_training_demonstrations": raise ValueError("Training-only calibration required")
    ckpt = Path("/data/jyc/data/VLA-Adapter/LIBERO-Spatial-Pro")
    policy = VLAAdapterPolicy(str(ckpt), "libero_spatial")
    set_seed_everywhere(42)
    try:
        policy.load()
        # Bounded pilot: seven Linear layers in the first language block.
        modules = {n:m for n,m in policy.model.named_modules()
                   if n.startswith("language_model.model.layers.0.") and isinstance(m, torch.nn.Linear)}
        if len(modules) != 7: raise RuntimeError(f"Unexpected pilot module count: {len(modules)}")
        captures = {}
        for split in ("calibration", "development"):
            with ActivationCollector(modules, max_rows=64, rows_per_call=8, seed=42) as collector:
                for row in manifest["observations"]:
                    if row["split"] != split: continue
                    path = args.observations / row["file"]
                    if sha(path) != row["sha256"]: raise ValueError("Calibration observation changed")
                    with np.load(path, allow_pickle=False) as archive:
                        raw = {k:archive[k] for k in archive.files}
                    with torch.inference_mode():
                        policy.act(build_observation(raw, row["frame_id"], manifest["instruction"]))
            captures[split] = collector.export()
        artifact = {}; report = []
        for name, layer in modules.items():
            result = search_input_scale(layer.weight, captures["calibration"][name])
            artifact[name] = result["scale"]
            x = captures["development"][name]["samples"].to(device=layer.weight.device)
            w = layer.weight.detach().float()
            with torch.no_grad():
                ref = F.linear(x, w)
                validation = []
                for scale in (torch.ones(w.shape[1]), result["scale"]):
                    scale = scale.to(w.device)
                    q,s,z = group_quantize_int4(w*scale[None],128)
                    wh = dequantize_int4(q,s.bfloat16(),z.bfloat16(),128)/scale[None]
                    validation.append(float((F.linear(x,wh)-ref).square().mean()))
            report.append(dict(layer=name, alpha=result["ratio"], calibration_mse=result["loss"],
                               calibration_rtn_mse=result["rtn_reference_loss"],
                               development_rtn_mse=validation[0], development_scaled_mse=validation[1],
                               calls=captures["calibration"][name]["calls"],
                               rows_seen=captures["calibration"][name]["rows_seen"]))
        args.out.mkdir(parents=True)
        torch.save(artifact,args.out/"input_scales.pt")
        torch.save(captures,args.out/"activation_samples.pt")
        payload = dict(implementation=IMPLEMENTATION, status="search_pilot_not_deployed_awq",
                       block_objective=False, clipping=False, observations_manifest_sha256=sha(manifest_path),
                       checkpoint=str(ckpt), model_weights_sha256=sha(ckpt/"model.safetensors"),
                       input_scales_sha256=sha(args.out/"input_scales.pt"),
                       scope="first_language_block_only", layers=report)
        (args.out/"report.json").write_text(json.dumps(payload,indent=2))
        print(json.dumps(payload,indent=2))
    finally:
        policy.unload()


if __name__ == "__main__": main()
