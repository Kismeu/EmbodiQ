"""Train-only AWQ block search -> saved artifact reload -> shared rollout pilot.

Only the first Qwen language block is quantized. This is not the full-scope AWQ
benchmark and must not be compared to the full-scope RTN success-rate row.
"""
import argparse
from contextlib import ExitStack
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise RuntimeError("Use a fresh pilot directory")
    args.out.mkdir(parents=True)
    import numpy as np
    import torch
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()
    sys.path.insert(0,str(ROOT.parent/"VLA-Adapter"))
    from embodiq.backends.adapter.policy import VLAAdapterPolicy
    from embodiq.backends.prismatic_env import PrismaticEnvFactory
    from embodiq.core.rollout import build_observation,run_episode
    from embodiq.core.protocol import PROTOCOLS
    from embodiq.core.result import TaskResult
    from embodiq.core.timing import LatencyAccumulator
    from embodiq.quant.calibration import ActivationCollector
    from embodiq.quant.awq_blocks import BlockCapture,search_block_scale,search_clip
    from embodiq.quant.awq_search import search_input_scale
    from embodiq.quant.awq_artifact import FORMAT,tensor_hash,load_and_apply
    obs_root=ROOT/"calib/observations_spatial_task0_v1"
    manifest_path=obs_root/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if manifest["source"]!="official_training_demonstrations":raise ValueError("Expected training demonstrations")
    checkpoint="/data/jyc/data/VLA-Adapter/LIBERO-Spatial-Pro"
    policy=VLAAdapterPolicy(checkpoint,"libero_spatial")
    try:
        policy.load()
        block=policy.model.language_model.model.layers[0]
        prefix="language_model.model.layers.0."
        modules={prefix+n:m for n,m in block.named_modules() if isinstance(m,torch.nn.Linear)}
        if len(modules)!=7:raise RuntimeError("Unexpected first-block structure")
        with ExitStack() as stack:
            collector=stack.enter_context(ActivationCollector(modules,max_rows=64))
            attention=stack.enter_context(BlockCapture(block.self_attn,limit=4))
            mlp=stack.enter_context(BlockCapture(block.mlp,limit=4))
            for row in manifest["observations"]:
                if row["split"]!="calibration":continue
                path=obs_root/row["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("Observation drift")
                with np.load(path,allow_pickle=False) as archive:raw={k:archive[k] for k in archive.files}
                with torch.inference_mode():policy.act(build_observation(raw,row["frame_id"],manifest["instruction"]))
        captures=collector.export()
        attn_result=search_block_scale(block.self_attn,[block.self_attn.q_proj,block.self_attn.k_proj,block.self_attn.v_proj],
                                      attention.records,captures[prefix+"self_attn.q_proj"]["mean_abs"])
        mlp_result=search_block_scale(block.mlp,[block.mlp.gate_proj,block.mlp.up_proj],mlp.records,
                                     captures[prefix+"mlp.gate_proj"]["mean_abs"])
        artifact=dict(format=FORMAT,group_size=128,implementation="awq_qwen_block_pilot_v1",layers={},
                      scope="first_language_block_only",not_for_full_scope_comparison=True,
                      observations_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        summaries=[]
        for name,layer in modules.items():
            if name.endswith(("q_proj","k_proj","v_proj")):scale=attn_result["scale"]
            elif name.endswith(("gate_proj","up_proj")):scale=mlp_result["scale"]
            elif name.endswith("down_proj"):scale=search_input_scale(layer.weight,captures[name])["scale"]
            else:scale=torch.ones(layer.in_features) # GQA output projection: no folding scale
            skip_clip=name.endswith(("q_proj","k_proj"))
            clip=None if skip_clip else search_clip(layer.weight,captures[name]["samples"],scale)["clip_max"]
            artifact["layers"][name]=dict(scale=scale,clip_max=clip,weight_sha256=tensor_hash(layer.weight),
                                          bias_sha256=tensor_hash(layer.bias),shape=list(layer.weight.shape))
            summaries.append(dict(layer=name,clip_applied=not skip_clip,shape=list(layer.weight.shape)))
        path=args.out/"awq_block.pt"
        torch.save(artifact,path)
        # Reload from the saved file; do not deploy in-memory search results.
        loaded=load_and_apply(policy.model,path)
        numerical=[]
        with torch.no_grad():
            for name in loaded["layers"]:
                module=policy.model.get_submodule(name)
                x=captures[name]["samples"].to("cuda:0",dtype=torch.bfloat16)
                actual=module(x).float()
                reference=torch.nn.functional.linear(x.float(),module.dequantized(),
                                                      module.bias.float() if module.bias is not None else None)
                rel=float((actual-reference).norm()/reference.norm().clamp(min=1e-8))
                if not torch.isfinite(actual).all() or rel>.05:raise ValueError(f"Packed numerical mismatch {name}: {rel}")
                numerical.append(dict(layer=name,relative_error=rel))
        report=dict(status="packed_block_pilot",scope="first_language_block_only",clipping=True,
                    attention_alpha=attn_result["alpha"],attention_loss=attn_result["loss"],
                    attention_rtn_loss=attn_result["rtn_loss"],mlp_alpha=mlp_result["alpha"],
                    mlp_loss=mlp_result["loss"],mlp_rtn_loss=mlp_result["rtn_loss"],
                    artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),layers=summaries,numerical=numerical)
        (args.out/"search_report.json").write_text(json.dumps(report,indent=2))
        factory=PrismaticEnvFactory()
        proto=dataclasses.replace(PROTOCOLS["unified.v3"],num_trials_per_task=2)
        env,description=factory.make("libero_spatial",0,resolution=256,seed=42)
        states=factory.initial_states("libero_spatial",0,2)
        policy.spec=dataclasses.replace(policy.spec,quant_id="awq-block-pilot")
        result=TaskResult(task_suite="libero_spatial",task_id=0,task_description=description,spec=policy.spec,
                          protocol_name=proto.name,protocol_fields={k:v for k,v in dataclasses.asdict(proto).items() if k!="name"},
                          gpu_id=3,quant=dict(method="awq_block_pilot",artifact_sha256=report["artifact_sha256"],
                                            scope="first_language_block_only",not_for_full_scope_comparison=True))
        acc=LatencyAccumulator()
        try:
            for ep in range(2):
                result.episodes.append(run_episode(env,policy,proto,suite="libero_spatial",task_id=0,episode_idx=ep,
                                       initial_state=states[ep],task_description=description,dummy_action=factory.dummy_action(),
                                       accumulator=acc))
        finally:env.close()
        result.latency=acc
        result.write(args.out)
        print(json.dumps(dict(successes=result.n_success,episodes=len(result.episodes),errors=result.n_error,
                              artifact=str(path),scope="first_language_block_only")))
        if result.n_error:raise RuntimeError("Pilot rollout contains errors")
    finally:policy.unload()


if __name__=="__main__":main()
