"""Download the stage-2 missing assets through a mirror, without user proxies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts" / "mirror_download"
WAN = ROOT.parent / "FastWAM" / "checkpoints"
JOBS = [
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", WAN / "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
     ["Wan2.2_VAE.safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"]),
    ("Wan-AI/Wan2.1-T2V-1.3B", WAN / "Wan-AI/Wan2.1-T2V-1.3B", ["google/umt5-xxl/*"]),
    ("openvla/openvla-7b-finetuned-libero-goal", Path("/data/jyc/models/openvla-7b-finetuned-libero-goal"),
     ["*.json", "*.model", "*.safetensors", "*.py"]),
    ("openvla/openvla-7b-finetuned-libero-10", Path("/data/jyc/models/openvla-7b-finetuned-libero-10"),
     ["*.json", "*.model", "*.safetensors", "*.py"]),
]


def clean_environment():
    for key in list(os.environ):
        if "proxy" in key.lower() or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            os.environ.pop(key, None)
    os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_OFFLINE="0", HF_HUB_ENABLE_HF_TRANSFER="0",
                      HF_HUB_DOWNLOAD_TIMEOUT="60", HF_HUB_ETAG_TIMEOUT="30")


def main():
    global STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--wan-original-only", action="store_true")
    args = parser.parse_args()
    jobs = JOBS
    if args.wan_original_only:
        STATE = STATE.with_name("mirror_download_wan_original")
        jobs = [("Wan-AI/Wan2.2-TI2V-5B", WAN / "Wan-AI/Wan2.2-TI2V-5B",
                 ["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"])]
    clean_environment()
    STATE.mkdir(parents=True, exist_ok=True)
    if args.background:
        # The worker holds an advisory lock too, so concurrent launches cannot
        # start duplicate transfers even if the status file is stale.
        with (STATE / "worker.log").open("ab", buffering=0) as log:
            command = [sys.executable, "-u", __file__]
            if args.wan_original_only:
                command.append("--wan-original-only")
            proc = subprocess.Popen(command, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True, env=os.environ.copy())
        print(json.dumps({"pid": proc.pid, "log": str(STATE / "worker.log")}))
        return

    import fcntl
    lock = (STATE / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another download worker is active.", flush=True)
        return
    import requests
    from huggingface_hub import HfApi, configure_http_backend, snapshot_download

    def backend():
        session = requests.Session()
        session.trust_env = False  # also disables .netrc and inherited proxies
        session.proxies.clear()
        return session

    configure_http_backend(backend_factory=backend)
    state = {"pid": os.getpid(), "endpoint": os.environ["HF_ENDPOINT"],
             "proxy_disabled": True, "status": "running", "jobs": []}

    def save():
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        tmp = STATE / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE / "status.json")

    save()
    for repo, dest, patterns in jobs:
        row = {"repo": repo, "destination": str(dest), "patterns": patterns, "status": "running"}
        state["jobs"].append(row)
        save()
        for attempt in range(1, 4):
            try:
                print(f"START {repo} attempt={attempt}", flush=True)
                info = HfApi(endpoint=os.environ["HF_ENDPOINT"], token=False).model_info(repo)
                row["revision"] = info.sha
                save()
                snapshot_download(repo, revision=info.sha, local_dir=str(dest), allow_patterns=patterns,
                                  endpoint=os.environ["HF_ENDPOINT"], token=False, max_workers=2)
                row["status"] = "downloaded"
                print(f"DONE {repo}", flush=True)
                break
            except Exception as exc:
                # Avoid logging signed URLs or environment values.
                row["attempt"] = attempt
                row["error_type"] = type(exc).__name__
                print(f"FAILED {repo} {type(exc).__name__}", flush=True)
                if attempt < 3:
                    time.sleep(15)
                else:
                    row["status"] = "failed"
            finally:
                save()
    state["status"] = "downloaded" if all(x["status"] == "downloaded" for x in state["jobs"]) else "partial_failure"
    save()


if __name__ == "__main__":
    main()
