"""Freeze working-tree code/evidence without committing unrelated user changes.

Archive manifest hashes refer to the exact bytes in the archive. Large external
checkpoints/data are referenced by the existing manifests, not silently copied
or claimed to have been rehashed. No network access is needed.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    files={}
    for name in ('src','tools','tests','docs','configs','env'):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.yaml','.yml','.sh','.txt','.sha256'):
                files[str(p.relative_to(ROOT))]=p
    for name in ('pyproject.toml','.gitignore'):
        files[name]=ROOT/name
    for tree in ('quant_stage5','quant_stage6'):
        for p in (ROOT/'experiments'/tree).rglob('*.py'):
            if '__pycache__' not in p.parts:files[str(p.relative_to(ROOT))]=p
    # Include all original per-task evidence needed to regenerate the main table.
    for directory in ('results/baseline_unified_v3_20260919/full','results/stage5_rtn_full',
                      'results/resource_bench','results/matrix','results/framework_closeout_20261009'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and p.suffix in ('.json','.md','.csv','.log','.txt','.png','.svg','.py'):
                files[str(p.relative_to(ROOT))]=p
    for name in ('episode_manifest.json','input_manifest.json','environment_manifest.json'):
        p=ROOT/'results/baseline_unified_v3_20260919'/name
        if p.is_file():files[str(p.relative_to(ROOT))]=p
    # Preserve local model-adapter source, including modifications outside git.
    upstream={}
    for repo,dirs in {'OpenVLA':['prismatic','experiments'],
                      'VLA-Adapter':['prismatic','experiments','vla-scripts'],
                      'FastWAM':['src','experiments','configs'],
                      'LIBERO':['libero']}.items():
        base=ROOT.parent/repo
        result=subprocess.run(['git','-C',str(base),'rev-parse','HEAD'],capture_output=True,text=True)
        upstream[repo]={'git_head':result.stdout.strip() if result.returncode==0 else None,
                        'included_source_dirs':dirs,'large_assets':'external, see input_manifest.json'}
        for pattern in ('LICENSE*', 'NOTICE*', 'COPYING*', 'requirements*.txt', 'pyproject.toml', 'setup.py', 'setup.cfg'):
            for p in base.glob(pattern):
                if p.is_file():files[f'upstream/{repo}/{p.name}']=p
        for directory in dirs:
            for p in (base/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.yml','.sh'):
                    files[f'upstream/{repo}/{p.relative_to(base)}']=p
    environments={}
    code="import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':sorted([(d.metadata.get('Name',''),d.version) for d in m.distributions()])}))"
    for env in ('openvla','vlaadapter','fastwam'):
        binary=f'/data/jyc/tools/anaconda3/envs/{env}/bin/python'
        proc=subprocess.run([binary,'-c',code],capture_output=True,text=True,check=True)
        environments[env]=json.loads(proc.stdout)
    archive=args.out/'source_and_evidence.tar.gz';manifest=[]
    with tarfile.open(archive,'w:gz') as tar:
        for name,path in sorted(files.items()):
            data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mtime=int(path.stat().st_mtime)
            info.mode=path.stat().st_mode & 0o777
            tar.addfile(info,io.BytesIO(data));manifest.append(dict(path=name,bytes=len(data),sha256=digest(data)))
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    metadata=dict(schema='embodiq/release-snapshot/1',created=time.strftime('%FT%T%z'),
                  git_head=head,working_tree_included=True,archive=archive.name,
                  archive_sha256=digest(archive.read_bytes()),files=manifest,
                  environments=environments,upstream=upstream,
                  external_inputs='Existing expected hashes retained; large checkpoint/data hashes were not revalidated in this freeze.')
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    (args.out/'SHA256SUMS').write_text(f"{metadata['archive_sha256']}  {archive.name}\n")
    # Read back every archived member, rather than trusting that tar writing worked.
    with tarfile.open(archive,'r:gz') as tar:
        for row in manifest:
            assert digest(tar.extractfile(row['path']).read())==row['sha256'],row['path']
    print(json.dumps(dict(status='verified',files=len(manifest),archive=str(archive),
                          archive_bytes=archive.stat().st_size,sha256=metadata['archive_sha256'])))


if __name__=='__main__':main()
