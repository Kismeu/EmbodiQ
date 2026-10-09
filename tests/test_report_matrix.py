"""Prevent plausible but wrong formal tables, using synthetic result fixtures."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import report_matrix as report
from embodiq.core.protocol import PROTOCOLS
from embodiq.core.admission import run_problems


def run(method='fp16',suite='libero_spatial',task=0):
    protocol=dataclasses.asdict(PROTOCOLS['unified.v3']);protocol['num_trials_per_task']=50
    q=dict(method=method,mode='none' if method=='fp16' else 'packed',scope=[] if method=='fp16' else ['llm'])
    if method!='fp16':q.update(w_bits=4,group_size=128,spec_hash='spec')
    return dict(schema_version='embodiq/run_result/2',task_suite=suite,task_id=task,
                policy=dict(family='openvla',model_id='openvla-7b',checkpoint=f'/checkpoints/{suite}',
                            chunk_size=1,default_replan_steps=1,torch_dtype='bfloat16',extra={}),
                quant=q,protocol=protocol,run=dict(contended=False,versions={}),
                episodes=[dict(episode_idx=i,seed=42,status='success',
                               aux=dict(initial_state_sha256=f'{task}-{i}')) for i in range(50)],
                metrics=dict(n_episodes=50,n_success=50,n_error=0,n_invalid=0,n_policy_error=0,n_timeout=0,
                             latency=dict(query_s=dict(mean=9,p95=10),per_env_step_s=9),
                             memory=dict(peak_alloc_bytes=99,nvml_peak_bytes=999)))


def formal_awq():
    d=run('awq');q=d['quant']
    q.update(implementation='public_block_v1',implementation_version='1')
    q['artifact']=dict(formal=True,validation_only=False,method='awq',model_family='openvla',
                       suite='libero_spatial',scope=['llm'],n_layers=224,
                       implementation='public_block_v1',implementation_version='1',
                       producer_code_sha256='a'*64,source_checkpoint_sha256='b'*64,
                       artifact_sha256='c'*64,calibration=dict(manifest_sha256='d'*64,n_calibration=32))
    return d


@pytest.mark.parametrize('change',[
    lambda d:d['quant']['artifact'].update(validation_only=True),
    lambda d:d['quant']['artifact'].update(formal=False),
    lambda d:d['quant']['artifact'].pop('formal'),
    lambda d:d['quant']['artifact'].update(n_layers=7),
    lambda d:d['quant']['artifact'].update(implementation='pipeline_validation_awq'),
    lambda d:d['quant'].update(scope=['vision_dino']),
    lambda d:d['quant'].update(mode='simulated'),
    lambda d:d['quant']['artifact']['calibration'].update(n_calibration=0),
])
def test_formal_admission_rejects_development_and_wrong_scope(change):
    d=formal_awq();assert not run_problems(d,model='openvla',method='awq',suite='libero_spatial')
    change(d)
    assert run_problems(d,model='openvla',method='awq',suite='libero_spatial')


def test_idm_cannot_enter_main_matrix():
    d=run();d['policy'].update(family='fastwam',model_id='fastwam-idm',extra=dict(action_infer_mode='idm'))
    assert any('first-frame' in e for e in run_problems(d,model='fastwam',method='fp16',suite='libero_spatial'))
    d['policy'].update(model_id='fastwam-uncond-first-frame',extra=dict(future_video=False))
    assert not run_problems(d,model='fastwam',method='fp16',suite='libero_spatial')


@pytest.mark.parametrize('mutation', ['duplicate','counts','seed','short'])
def test_bad_episode_data_prevents_complete_cell(mutation):
    runs={i:run(task=i) for i in range(10)}
    d=runs[0]
    if mutation=='duplicate':d['episodes'][1]=copy.deepcopy(d['episodes'][0])
    if mutation=='counts':d['metrics']['n_success']=49
    if mutation=='seed':d['episodes'][0]['seed']=7
    if mutation=='short':d['episodes'].pop()
    problems=[];cell=report.build_cell('openvla','fp16','libero_spatial',runs,problems)
    assert problems and not cell['complete']


@pytest.mark.parametrize('kind',['implementation','checkpoint','duplicate_task'])
def test_mixed_identity_is_rejected(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'METHOD_SOURCES',{'awq':('awq',)})
    directory=tmp_path/'awq';directory.mkdir()
    for i in range(10):
        d=formal_awq();d['task_id']=i
        if i==9 and kind=='implementation':
            d['quant']['implementation']='different';d['quant']['artifact']['implementation']='different'
        if i==9 and kind=='checkpoint':d['policy']['checkpoint']='/different'
        (directory/f'{i}_results.json').write_text(json.dumps(d))
    if kind=='duplicate_task':(directory/'duplicate_results.json').write_text(json.dumps(formal_awq()))
    cells,problems=report.load_cells()
    assert not cells and problems


def resource_fixture(tmp_path):
    problems=[]
    cells={('openvla','fp16',s):report.build_cell('openvla','fp16',s,
           {i:run(suite=s,task=i) for i in range(10)},problems) for s in report.SUITES}
    assert not problems
    tasks=[];digest=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest()
    (tmp_path/'inputs_v1.json').write_text(json.dumps(dict(tasks=tasks,manifest_sha256=digest)))
    bench=dict(schema='embodiq/resource_bench/1',phase='complete',model='openvla',quant='fp16',
               contended=False,warmup_queries=20,measured_passes=3,inputs_manifest_sha256=digest,suites={})
    for s in report.SUITES:
        bench['suites'][s]=dict(checkpoint=f'/checkpoints/{s}',n_inputs=50,n_queries_measured=150,passes=3,
                               chunk_size=1,replan_steps=1,policy=dict(model_id='openvla-7b',torch_dtype='bfloat16'),
                               quant=dict(method='fp16',scope=[]),latency=dict(query_s=dict(mean=.2),per_env_step_s=.2),
                               memory=dict(peak_alloc_bytes=10,nvml_peak_bytes=20,load_peak_alloc_bytes=30,weights_bytes=5))
    return cells,bench


def test_resources_replace_rollout_and_keep_provenance(tmp_path):
    cells,b=resource_fixture(tmp_path)
    (tmp_path/'openvla_fp16.json').write_text(json.dumps(b))
    configs=report.build_configs(cells);problems=[]
    report.attach_resource_benches(configs,cells,problems,root=tmp_path)
    row=next(r for r in configs if (r['model'],r['method'])==('openvla','fp16'))
    assert not problems and row['memory']['nvml_peak_bytes']==20
    assert row['rollout_memory']['nvml_peak_bytes']==999
    assert row['latency']['query_s_mean']==.2 and row['resource_bench']['sha256']
    assert cells[('openvla','fp16','libero_spatial')]['resource_bench']['status']=='complete'


@pytest.mark.parametrize('mutation',['missing','contended','checkpoint','manifest','incomplete','nan'])
def test_unusable_resource_never_falls_back_to_rollout(tmp_path,mutation):
    cells,b=resource_fixture(tmp_path)
    if mutation=='contended':b['contended']=True
    if mutation=='checkpoint':b['suites']['libero_goal']['checkpoint']='/wrong'
    if mutation=='manifest':b['inputs_manifest_sha256']='wrong'
    if mutation=='incomplete':b['suites'].pop('libero_goal')
    if mutation=='nan':b['suites']['libero_goal']['memory']['nvml_peak_bytes']=float('nan')
    if mutation!='missing':(tmp_path/'openvla_fp16.json').write_text(json.dumps(b))
    configs=report.build_configs(cells);problems=[]
    report.attach_resource_benches(configs,cells,problems,root=tmp_path)
    row=next(r for r in configs if (r['model'],r['method'])==('openvla','fp16'))
    assert row['memory']=={} and row['latency']['query_s_mean'] is None
    assert row['resource_bench']['status']==('missing' if mutation=='missing' else 'rejected')


def test_episode_manifest_rejects_unpaired_states(tmp_path):
    cells,_=resource_fixture(tmp_path)
    tasks=[dict(suite=s,task_id=t,episodes=[dict(seed=42,state_sha256=f'{t}-{i}') for i in range(50)])
           for s in report.SUITES for t in range(10)]
    path=tmp_path/'episodes.json';path.write_text(json.dumps(dict(protocol='unified.v3',tasks=tasks)))
    problems=[];report.validate_episode_manifest(cells,problems,path)
    assert not problems and len(cells)==4
    cell=cells[('openvla','fp16','libero_spatial')]
    cell['_episodes']['0:42:unrecognized-state']=True
    report.validate_episode_manifest(cells,problems,path)
    assert problems and ('openvla','fp16','libero_spatial') not in cells
