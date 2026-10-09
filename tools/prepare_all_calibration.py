"""Render downloaded training tasks sequentially; wait for remaining files."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calib/observations_training_v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (OUT/"controller.log").open("ab", buffering=0) as log:
            p=subprocess.Popen([sys.executable,"-u",__file__],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({"pid":p.pid,"status":str(OUT/"status.json")})); return
    lock=(OUT/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/"status.json").exists(): raise RuntimeError("Existing preparation run; inspect before resuming")
    from libero.libero import benchmark
    jobs=[]
    for suite in SUITES:
        tasks=benchmark.get_benchmark_dict()[suite]()
        for i in range(tasks.n_tasks):
            task=tasks.get_task(i)
            jobs.append(dict(suite=suite,task_id=i,file=f"{suite}/{task.name}_demo.hdf5",status="waiting"))
    state=dict(pid=os.getpid(),phase="preparing",jobs=jobs)
    def save():
        state["updated"]=time.strftime("%FT%T%z")
        temp=OUT/"status.tmp";temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/"status.json")
    save(); deadline=time.monotonic()+24*3600
    while any(j["status"]=="waiting" for j in jobs):
        if time.monotonic()>deadline:
            state["phase"]="download_wait_timeout";save();return
        downloaded=json.loads((ROOT/"calib/libero_training_demos/status.json").read_text())
        ready={r["name"] for r in downloaded["files"] if r["status"]=="downloaded"}
        progressed=False
        for row in jobs:
            if row["status"]!="waiting" or row["file"] not in ready: continue
            progressed=True;row["status"]="rendering";save()
            dest=OUT/row["suite"]/f"task{row['task_id']}"
            log=OUT/f"{row['suite']}_task{row['task_id']}.log"
            cmd=[sys.executable,"-u",str(ROOT/"tools/prepare_calibration_observations.py"),
                 "--suite",row["suite"],"--task",str(row["task_id"]),"--out",str(dest)]
            with log.open("w") as f:
                rc=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
            row.update(status="complete" if rc==0 and (dest/"manifest.json").exists() else "failed",exit_code=rc)
            save()
        if downloaded["phase"] in ("downloaded","partial_failure") and not progressed:
            for row in jobs:
                if row["status"]=="waiting":row["status"]="missing_download"
            break
        if not progressed:time.sleep(20)
    state["phase"]="complete" if all(j["status"]=="complete" for j in jobs) else "partial_failure"
    state["calibration_observations"]=32*sum(j["status"]=="complete" for j in jobs)
    state["development_observations"]=8*sum(j["status"]=="complete" for j in jobs)
    save()


if __name__=="__main__":main()
