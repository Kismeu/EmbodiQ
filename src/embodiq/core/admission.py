"""CPU-only identity and provenance gates for formal task-1.4 reports."""
from __future__ import annotations
import json
import math
from embodiq.config import _SCOPE_KEYS


def artifact_problems(meta: dict, *, method: str, model: str, suite: str) -> list[str]:
    errors = []
    if not meta:
        return ['missing artifact provenance']
    if meta.get('formal') is not True or meta.get('validation_only') is not False:
        errors.append('artifact must explicitly declare formal=true and validation_only=false')
    if any(word in str(meta.get('implementation','')).lower() for word in ('pilot','validation','pending','experimental')):
        errors.append('development implementation cannot enter the formal matrix')
    for key in ('implementation','implementation_version','producer_code_sha256',
                'source_checkpoint_sha256','artifact_sha256'):
        if not meta.get(key): errors.append(f'missing artifact {key}')
    if meta.get('method') != method or meta.get('model_family') != model or meta.get('suite') != suite:
        errors.append('artifact method/model/suite mismatch')
    if set(meta.get('scope',[])) != _SCOPE_KEYS.get(model,set()):
        errors.append('artifact scope mismatch')
    calibration=meta.get('calibration') or {}
    if not calibration.get('manifest_sha256') or calibration.get('n_calibration',0)<=0:
        errors.append('artifact has no real calibration provenance')
    expected={'openvla':224,'vla_adapter':568,'fastwam':600}.get(model)
    if meta.get('n_layers') != expected: errors.append('artifact does not cover full INT4 scope')
    return errors


def condition_identity(policy: dict, quant: dict, protocol: dict | None = None) -> str:
    """Comparison condition; checkpoint/artifact may legitimately vary by suite."""
    extra=policy.get('extra') or {}; artifact=quant.get('artifact') or {}
    return json.dumps(dict(
        policy={k:policy.get(k) for k in ('family','model_id','cameras','env_resolution','uses_proprio',
                                         'chunk_size','default_replan_steps','torch_dtype','gripper')},
        runtime={k:extra.get(k) for k in ('action_infer_mode','sigma_shift','num_inference_steps',
                                        'compile_action_infer','input_hw')},
        quant={k:quant.get(k) for k in ('method','mode','w_bits','group_size','spec_hash','hsa')},
        scope=sorted(quant.get('scope',[])),
        implementation=quant.get('implementation',artifact.get('implementation')),
        implementation_version=quant.get('implementation_version',artifact.get('implementation_version')),
        protocol=protocol),sort_keys=True)


def run_problems(d: dict, *, model: str, method: str, suite: str) -> list[str]:
    errors=[];q=d.get('quant') or {};p=d.get('policy') or {};proto=d.get('protocol') or {}
    for obj in (d,q,p):
        if obj.get('validation_only') or obj.get('formal') is False or obj.get('not_for_full_scope_comparison'):
            errors.append('development-only result')
    if q.get('method')!=method or p.get('family')!=model or d.get('task_suite')!=suite:
        errors.append('run method/model/suite mismatch')
    expected_scope=set() if method=='fp16' else _SCOPE_KEYS.get(model,set())
    if set(q.get('scope',[]))!=expected_scope:errors.append('quantization scope differs from formal model scope')
    if method!='fp16' and (q.get('mode')!='packed' or q.get('w_bits')!=4 or q.get('group_size')!=128):
        errors.append('formal quantization requires packed W4 group128')
    if proto.get('name')!='unified.v3' or proto.get('num_trials_per_task')!=50:
        errors.append('formal runs require unified.v3 and 50 trials per task')
    if proto.get('seed_base')!=42 or proto.get('num_steps_wait')!=30 or proto.get('env_resolution')!=256:
        errors.append('formal protocol seed/wait/resolution mismatch')
    if proto.get('max_steps')!={'libero_spatial':400,'libero_object':400,'libero_goal':400,'libero_10':700}:
        errors.append('formal protocol action-step limits mismatch')
    if model=='fastwam' and policy_mode(p)!='first_frame':
        errors.append('IDM/unknown mode cannot enter the first-frame main matrix')
    if method in ('awq','hsa'):
        art=q.get('artifact') or {}
        errors.extend(artifact_problems(art,method=method,model=model,suite=suite))
        for key in ('implementation','implementation_version'):
            if q.get(key)!=art.get(key):errors.append(f'artifact/run {key} mismatch')
    return errors


def finite_positive(value) -> bool:
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>0


def policy_mode(policy: dict) -> str | None:
    extra=policy.get('extra') or {}
    mode=extra.get('action_infer_mode')
    if mode is not None:return mode
    # Explicit legacy identity from the frozen, verified first-frame runs.
    if policy.get('model_id')=='fastwam-uncond-first-frame' and extra.get('future_video') is False:
        return 'first_frame'
    return None
