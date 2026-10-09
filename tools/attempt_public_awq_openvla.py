"""Bounded public-AWQ experiment, isolated from the formal result matrix.

Reuse Han Lab's block search functions with the existing tinygemm quantizer.
The upstream CUDA engine and scale folding are deliberately not used: the
evaluator retains explicit input scales. All departures are saved in report.json.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_search(upstream):
    """Load only public search functions; no fake CUDA extension is installed."""
    import torch
    from transformers.models.bloom.modeling_bloom import BloomBlock
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from embodiq.quant.pack import group_quantize_int4, dequantize_int4

    def quantize(w, n_bit=4, zero_point=True, q_group_size=128, **kwargs):
        if n_bit != 4 or not zero_point or q_group_size != 128 or kwargs:
            raise ValueError("Unsupported quantizer configuration")
        shape = w.shape
        q, s, z = group_quantize_int4(w.reshape(-1, shape[-1]), 128)
        return dequantize_int4(q, s.bfloat16(), z.bfloat16(), 128).reshape(shape).to(w.dtype)

    def get_op_name(module, op):
        return next(name for name, child in module.named_modules() if child is op)

    ns = dict(torch=torch, nn=torch.nn, gc=__import__('gc'),
              BloomBlock=BloomBlock, OPTDecoderLayer=OPTDecoderLayer,
              LlamaDecoderLayer=LlamaDecoderLayer, Qwen2DecoderLayer=Qwen2DecoderLayer,
              get_op_name=get_op_name, pseudo_quantize_tensor=quantize)

    class BindQuantizer(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == 'quantizer' and node.level == 1:
                return None
            return node

    for filename, names in [('auto_scale.py', {'get_act_scale', 'auto_scale_block'}),
                            ('auto_clip.py', {'auto_clip_layer'})]:
        path = upstream / 'awq/quantize' / filename
        tree = ast.parse(path.read_text())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in body} != names:
            raise RuntimeError('Upstream function layout changed')
        tree = BindQuantizer().visit(ast.Module(body=body, type_ignores=[]))
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return ns['auto_scale_block'], ns['auto_clip_layer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--blocks', type=int, default=32)
    ap.add_argument('--calibration', type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.blocks <= 32 or not 1 <= args.calibration <= 32:
        raise ValueError('Invalid bounded pilot size')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL='egl',
                      PYOPENGL_PLATFORM='egl', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      QVLA_TF_CPU_ONLY='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(ROOT/'src'))
    sys.path.insert(0, str(ROOT.parent/'OpenVLA'))
    from embodiq.backends.compat import tf_cpu_guard
    tf_cpu_guard.apply()
    import numpy as np
    import torch
    from embodiq.backends.openvla.policy import OpenVLAPolicy
    from embodiq.core.rollout import build_observation
    from embodiq.quant.artifact import tensor_sha256, FORMAT
    from embodiq.quant.qlinear import QuantLinear
    from embodiq.quant.swap import INT4_REL_ERR_BAND
    from embodiq.quant.awq_blocks import map_tensors

    torch.manual_seed(42)
    torch.set_num_threads(4)
    report = dict(status='running', formal=False, model='openvla', suite='libero_spatial',
                  started=time.strftime('%FT%T%z'), blocks_requested=args.blocks,
                  gpu=args.gpu, calibration_observations=args.calibration,
                  upstream_commit=subprocess.check_output(
                      ['git','-C',str(args.upstream),'rev-parse','HEAD'], text=True).strip(),
                  producer_sha256=sha(__file__), upstream_sha256={
                      n:sha(args.upstream/'awq/quantize'/n) for n in ('auto_scale.py','auto_clip.py')},
                  adaptations=['public block-search functions loaded without unused CUDA extension',
                               'pseudo quantizer bound to existing tinygemm BF16 scale/offset convention',
                               'explicit per-Linear input scales instead of norm/linear folding',
                               'prefill-only calibration; no cached decoding replay',
                               'bounded training subset; never a formal success-rate result'],
                  thresholds=dict(weight_relative_error_max=INT4_REL_ERR_BAND[1],
                                  packed_output_relative_error_max=0.05), blocks=[])

    def save():
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))

    policy = None
    try:
        save()
        scale_search, clip_search = load_search(args.upstream)
        obs_dir=ROOT/'calib/observations_spatial_task0_v1'
        mp=obs_dir/'manifest.json'; manifest=json.loads(mp.read_text())
        report['calibration_manifest_sha256']=sha(mp)
        rows=[r for r in manifest['observations'] if r['split']=='calibration'][:args.calibration]
        observations=[]
        for row in rows:
            p=obs_dir/row['file']
            if sha(p)!=row['sha256']: raise ValueError('Observation hash mismatch')
            with np.load(p,allow_pickle=False) as z: raw={k:z[k] for k in z.files}
            observations.append(build_observation(raw,row['frame_id'],manifest['instruction']))
        checkpoint='/data/jyc/models/openvla-7b-finetuned-libero-spatial'
        policy=OpenVLAPolicy(checkpoint,'libero_spatial'); policy.load()
        print('Loaded OpenVLA; beginning public block search',flush=True)
        layers={}
        for bi,block in enumerate(policy.model.language_model.model.layers[:args.blocks]):
            block_start=time.monotonic()
            modules={n:m for n,m in block.named_modules() if isinstance(m,torch.nn.Linear)}
            features={n:[] for n in modules}; attn_kwargs=[]; handles=[]
            def hook_for(name):
                def hook(module,inputs):
                    x=inputs[0]
                    if x.shape[-2]>1:
                        features[name].append(x.detach().cpu().clone())
                return hook
            def capture_attention(module,inputs,kwargs):
                x=kwargs.get('hidden_states', inputs[0] if inputs else None)
                if x.shape[-2]<=1:return
                kw={k:v for k,v in kwargs.items() if k!='hidden_states'}
                cache=kw.get('past_key_value')
                if cache is not None and cache.get_seq_length(bi)!=0:
                    raise ValueError('Unexpected nonempty prefill cache')
                kw['past_key_value']=None; kw['use_cache']=False
                attn_kwargs.append(map_tensors(kw,'cpu'))
            try:
                handles=[m.register_forward_pre_hook(hook_for(n)) for n,m in modules.items()]
                handles.append(block.self_attn.register_forward_pre_hook(capture_attention,with_kwargs=True))
                with torch.inference_mode():
                    for obs in observations: policy.act(obs)
            finally:
                for h in handles:h.remove()
            if any(len(xs)!=len(rows) for xs in features.values()) or len(attn_kwargs)!=len(rows):
                raise RuntimeError('Unexpected prefill capture count')
            feats={n:torch.cat(xs,dim=0) for n,xs in features.items()}
            kw=map_tensors(attn_kwargs[0],'cuda')
            # All observations share one task/prompt and therefore equal lengths.
            for key in ('position_ids','attention_mask','cache_position'):
                vals=[k.get(key) for k in attn_kwargs]
                if vals[0] is not None and not all(torch.equal(vals[0],v) for v in vals):
                    raise RuntimeError('Prefill kwargs differ across calibration observations')
            originals={n:m.weight.detach().clone() for n,m in modules.items()}
            scales=scale_search(block,kw,4,dict(zero_point=True,q_group_size=128),feats)
            scale_map={n:s for _,names,s in scales for n in names}
            block_report=dict(index=bi,n_prefill=len(rows),scale_groups=[list(ns) for _,ns,_ in scales],layers=[])
            report['blocks'].append(block_report)
            failed=[]
            for name,linear in modules.items():
                # Public search must restore weights; explicit scales act on originals.
                if not torch.equal(linear.weight,originals[name]):
                    raise RuntimeError(f'Public search mutated original weights: {name}')
                s=scale_map.get(name,torch.ones(linear.in_features)).float().to('cuda')
                x=feats[name].reshape(-1,linear.in_features).to('cuda')
                # Bound clipping activation storage, retaining true captured token rows.
                idx=torch.linspace(0,len(x)-1,min(64,len(x)),device=x.device).long()
                sample=x.index_select(0,idx)
                clip=None
                if not name.endswith(('q_proj','k_proj')):
                    clip=clip_search(linear.weight.float()*s[None],sample.float()/s[None],
                                     4,dict(zero_point=True,q_group_size=128),n_sample_token=len(sample)).squeeze(-1)
                with torch.no_grad():
                    packed=QuantLinear.from_linear(linear,input_scale=s,clip_max=clip)
                    reconstructed=packed.dequantized()
                    weight_rel=float((reconstructed-originals[name].float()).norm()/originals[name].float().norm())
                    actual=packed(sample).float()
                    expected=torch.nn.functional.linear(sample.float(),reconstructed,
                                linear.bias.float() if linear.bias is not None else None)
                    packed_rel=float((actual-expected).norm()/expected.norm().clamp(min=1e-8))
                    original_output=torch.nn.functional.linear(sample.float(),originals[name].float(),
                                linear.bias.float() if linear.bias is not None else None)
                    output_rel=float((actual-original_output).norm()/original_output.norm().clamp(min=1e-8))
                    ok=bool(torch.isfinite(actual).all()) and INT4_REL_ERR_BAND[0]<=weight_rel<=INT4_REL_ERR_BAND[1] and packed_rel<=0.05
                rec=dict(name=name,weight_relative_error=weight_rel,packed_relative_error=packed_rel,
                         calibration_output_relative_error=output_rel,
                         scale_min=float(s.min()),scale_max=float(s.max()),clipped=clip is not None,passed=ok)
                block_report['layers'].append(rec)
                full=f'language_model.model.layers.{bi}.{name}'
                layers[full]=dict(weight_sha256=tensor_sha256(linear.weight),bias_sha256=tensor_sha256(linear.bias),
                                  input_scale=s.cpu(),clip_max=clip.cpu() if clip is not None else None)
                if not ok:failed.append(rec)
                print(f'block={bi} {name} weight_rel={weight_rel:.6f} packed_rel={packed_rel:.6f} passed={ok}',flush=True)
                del packed,reconstructed,actual,expected,original_output,x,sample
            block_report['elapsed_s']=time.monotonic()-block_start
            torch.save(dict(format='embodiq/rejected-awq-diagnostics/1',formal=False,
                            validation_only=True,layers=layers),args.out/'candidate_layers.pt')
            save()
            if failed:raise RuntimeError(f'Existing numerical gates rejected block {bi}: {failed}')
            del originals,features,feats
            torch.cuda.empty_cache()
        payload=dict(format=FORMAT,method='awq',implementation='hanlab_block_explicit_tinygemm_pilot_v1',
                     implementation_version='1',producer_code_sha256=sha(__file__),model_family='openvla',
                     suite='libero_spatial',w_bit=4,group_size=128,scope=['llm'],
                     source_checkpoint_sha256=hashlib.sha256(json.dumps({n:r['weight_sha256'] for n,r in layers.items()},sort_keys=True).encode()).hexdigest(),
                     source_checkpoint_hash_kind='selected_weight_tensor_digest',
                     calibration=dict(manifest_sha256=sha(mp),n_calibration=len(rows),n_development=0,
                                      source='official training observations, bounded pilot'),
                     layers=layers,formal=False,validation_only=True)
        torch.save(payload,args.out/'artifact.pt')
        report.update(status='numerical_pass_pilot_only',n_layers=len(layers),artifact_sha256=sha(args.out/'artifact.pt'))
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        (args.out/'traceback.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if policy is not None: policy.unload()
        report['finished']=time.strftime('%FT%T%z');save()
    return 0 if report['status']=='numerical_pass_pilot_only' else 1


if __name__=='__main__':
    raise SystemExit(main())
