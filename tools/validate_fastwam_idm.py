"""Real optional-IDM query and one shared-runner episode, outside the main matrix."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL='egl',
                      PYOPENGL_PLATFORM='egl', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      DIFFSYNTH_SKIP_DOWNLOAD='true', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(ROOT/'src'))
    from embodiq.backends.fastwam.policy import FastWAMPolicy
    from embodiq.backends.fastwam.env import FastWAMEnvFactory
    from embodiq.core.rollout import build_observation, run_episode
    from embodiq.core.protocol import PROTOCOLS
    from embodiq.core.result import TaskResult
    from embodiq.core.timing import LatencyAccumulator
    import numpy as np
    import torch

    repo=ROOT.parent/'FastWAM'; release=repo/'checkpoints/fastwam_release'
    checkpoint=release/'libero_optional_idm_2cam224.pt'
    stats=release/'libero_optional_idm_2cam224_dataset_stats.json'
    report=dict(status='running',formal=False,mode='idm',gpu=args.gpu,
                checkpoint=str(checkpoint),started=time.strftime('%FT%T%z'),
                torch=torch.__version__,cuda=torch.version.cuda,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                video_denoise_calls=0,latent_shapes=[])
    (args.out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    policy=env=None
    def save(): (args.out/'validation.json').write_text(json.dumps(report,indent=2))
    try:
        save()
        policy=FastWAMPolicy(checkpoint,'libero_spatial',repo_root=repo,
                            dataset_stats=stats,action_infer_mode='idm')
        policy.load()
        report['policy']=dataclasses.asdict(policy.spec)
        report['policy']['gripper']=policy.spec.gripper.value
        original=policy.model._denoise_video
        def observed(*a,**kw):
            shape=list(kw['latents_video'].shape)
            report['video_denoise_calls']+=1
            if shape not in report['latent_shapes']: report['latent_shapes'].append(shape)
            return original(*a,**kw)
        policy.model._denoise_video=observed
        obsroot=ROOT/'calib/observations_spatial_task0_v1'
        manifest=json.loads((obsroot/'manifest.json').read_text())
        row=next(r for r in manifest['observations'] if r['split']=='development')
        p=obsroot/row['file']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Observation changed')
        with np.load(p,allow_pickle=False) as z: raw={k:z[k] for k in z.files}
        policy.reset(task_id=0,episode_idx=0,task_description=manifest['instruction'],seed=42)
        chunk=policy.act(build_observation(raw,row['frame_id'],manifest['instruction']))
        if chunk.actions.shape!=(32,7) or chunk.num_valid!=10 or not np.isfinite(chunk.actions).all():
            raise RuntimeError('Invalid IDM action chunk')
        if report['video_denoise_calls']==0 or not any(s[2]>1 for s in report['latent_shapes']):
            raise RuntimeError('Future-video denoising was not executed')
        report.update(query_passed=True,query_video_denoise_calls=report['video_denoise_calls'],
                      action_shape=list(chunk.actions.shape),num_valid=chunk.num_valid)
        save();print('IDM real query passed; starting one unified.v3 episode',flush=True)
        factory=FastWAMEnvFactory();proto=dataclasses.replace(PROTOCOLS['unified.v3'],num_trials_per_task=1)
        env,desc=factory.make('libero_spatial',0,resolution=256,seed=42)
        states=factory.initial_states('libero_spatial',0,1)
        acc=LatencyAccumulator()
        ep=run_episode(env,policy,proto,suite='libero_spatial',task_id=0,episode_idx=0,
                       initial_state=states[0],task_description=desc,dummy_action=factory.dummy_action(),accumulator=acc)
        result=TaskResult(task_suite='libero_spatial',task_id=0,task_description=desc,
                          spec=policy.spec,protocol_name=proto.name,
                          protocol_fields={k:v for k,v in dataclasses.asdict(proto).items() if k!='name'},
                          gpu_id=args.gpu,quant=dict(method='fp16',mode='none',scope=[],formal=False,
                                                  validation_only=True))
        result.episodes.append(ep);result.latency=acc;result.memory=policy.memory_footprint();result.write(args.out)
        report.update(episode_status=ep.status.value,episode_env_steps=ep.env_steps,episode_queries=ep.n_queries)
        if ep.status.value not in ('success','timeout'):raise RuntimeError(f'IDM rollout error: {ep.error}')
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        (args.out/'traceback.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        if env is not None:env.close()
        if policy is not None:policy.unload()
        report['finished']=time.strftime('%FT%T%z');save()
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
