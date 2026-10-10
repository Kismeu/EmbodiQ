"""Generate a full-scope AWQ artifact from the prepared LIBERO observations.

This uses the repository's explicit activation-aware per-Linear search. It is a
real calibration artifact, but its metadata names the implementation so it is
not confused with Han-Lab's block optimizer. The evaluator still validates every
layer/hash/scope before deployment.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SUITES=("libero_spatial","libero_object","libero_goal","libero_10")
CKPTS={"libero_spatial":"LIBERO-Spatial-Pro","libero_object":"LIBERO-Object-Pro","libero_goal":"LIBERO-Goal-Pro","libero_10":"LIBERO-Long-Pro"}

def sha_file(p: Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model',required=True,choices=('vla_adapter','openvla','fastwam'))
 ap.add_argument('--suite',required=True,choices=SUITES)
 ap.add_argument('--checkpoint',type=Path,required=True)
 ap.add_argument('--repo-root',type=Path,required=True)
 ap.add_argument('--observations',type=Path,default=ROOT/'calib/observations_training_v1')
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--gpu',type=int,default=0)
 ap.add_argument('--scopes',default='llm,vision_dino,vision_siglip,projector,action_head', help='comma-separated quantization scopes')
 ap.add_argument('--limit',type=int,default=0,help='development sample limit; output is marked non-formal')
 args=ap.parse_args()
 if args.out.exists(): raise SystemExit(f'output exists: {args.out}')
 os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',QVLA_TF_CPU_ONLY='1')
 sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(args.repo_root.resolve()))
 import numpy as np, torch
 from embodiq.backends.compat import tf_cpu_guard; tf_cpu_guard.apply()
 from embodiq.core.rollout import build_observation
 from embodiq.quant.calibration import ActivationCollector
 from embodiq.quant.awq_search import search_input_scale, IMPLEMENTATION
 from embodiq.quant.awq_blocks import search_clip
 from embodiq.quant.artifact import FORMAT, tensor_sha256
 from embodiq.quant.spec import QuantSpec
 from embodiq.quant.walker import plan_quantization
 from embodiq.backends.adapter.policy import VLAAdapterPolicy
 if args.model != 'vla_adapter': raise SystemExit('当前生成器先支持 vla_adapter；OpenVLA/FastWAM 使用同一契约但需各自输入预处理适配')
 policy=VLAAdapterPolicy(str(args.checkpoint),args.suite,device_index=0)
 manifests=[]; rows=[]
 for task in range(10):
  mp=args.observations/args.suite/f'task{task}'/'manifest.json'
  m=json.loads(mp.read_text()); manifests.append(mp)
  for o in m['observations']:
   if o['split']=='calibration': rows.append((mp,m,o))
 if args.limit: rows=rows[:args.limit]
 try:
  policy.load(); roots=policy.quantization_roots()
  spec=QuantSpec(method='awq',w_bit=4,group_size=128)
  scopes={x.strip() for x in args.scopes.split(',') if x.strip()}
  report=plan_quantization(roots,args.model,spec,enabled_scopes=scopes)
  targets=[r for r in report.layers if r.decision=='int4']
  modules={}
  for rec in targets:
   root=roots[rec.root]; local=rec.name.split('.',1)[1] if len(roots)>1 else rec.name
   modules[rec.name]=root.get_submodule(local)
  print(f'collecting {len(rows)} observations for {len(modules)} int4 layers',flush=True)
  with ActivationCollector(modules,max_rows=64,rows_per_call=8,seed=42) as c:
   for i,(mp,m,o) in enumerate(rows,1):
    fp=mp.parent/o['file']
    if sha_file(fp)!=o['sha256']: raise ValueError(f'observation drift: {fp}')
    with np.load(fp,allow_pickle=False) as z: raw={k:z[k] for k in z.files}
    with torch.inference_mode(): policy.act(build_observation(raw,o['frame_id'],m['instruction']))
    if i%50==0: print(f'  {i}/{len(rows)}',flush=True)
  caps=c.export(allow_missing=True); missing=[n for n in modules if n not in caps]; print(f'unobserved layers={len(missing)}',flush=True); layers={}
  for i,rec in enumerate(targets,1):
   layer=modules[rec.name]
   clip=None
   if rec.name in caps:
    result=search_input_scale(layer.weight,caps[rec.name],group_size=128); scale=result['scale']
    if 'qkv' in rec.name:
     clip=search_clip(layer.weight,caps[rec.name]['samples'],scale,group_size=128)['clip_max']
   else:
    # Unreachable Linear leaves cannot have an activation statistic. They are
    # still covered for hash/scope completeness using neutral AWQ scaling; the
    # artifact records them explicitly and they do not affect inference.
    scale=torch.ones(layer.in_features,dtype=torch.float32)
   layers[rec.name]={'weight_sha256':tensor_sha256(layer.weight),'bias_sha256':tensor_sha256(layer.bias),'input_scale':scale,'clip_max':clip}
   if i%50==0: print(f'  searched {i}/{len(targets)}',flush=True)
  ckpt_hash=hashlib.sha256(json.dumps({p.name:sha_file(p) for p in sorted(args.checkpoint.rglob('*')) if p.is_file()},sort_keys=True).encode()).hexdigest()
  cal_hash=hashlib.sha256(''.join(sha_file(p) for p in manifests).encode()).hexdigest()
  payload={'format':FORMAT,'method':'awq','implementation':IMPLEMENTATION,'implementation_version':'1','producer_code_sha256':sha_file(ROOT/'tools/generate_awq_artifact.py'),'model_family':args.model,'suite':args.suite,'w_bit':4,'group_size':128,'source_checkpoint_sha256':ckpt_hash,'calibration':{'manifest_sha256':cal_hash,'n_calibration':len(rows),'n_development':0,'source':str(args.observations)},'scope':sorted(scopes),'layers':layers,'created':time.strftime('%FT%T%z'),'formal':not bool(args.limit),'unobserved_layers':missing}
  args.out.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.out)
  print(f'wrote {args.out} layers={len(layers)} formal={not bool(args.limit)}',flush=True)
 finally: policy.unload()
if __name__=='__main__': main()
