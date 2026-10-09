"""Pinned official LIBERO training demonstrations through a proxy-free mirror."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from download_missing_models import clean_environment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calib/libero_training_demos"
REPO = "yifengzhu-hf/LIBERO-datasets"
REV = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    clean_environment()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (OUT / "download.log").open("ab", buffering=0) as log:
            p = subprocess.Popen([sys.executable, "-u", __file__], stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, env=os.environ.copy())
        print(json.dumps({"pid": p.pid, "status": str(OUT / "status.json")})); return
    lock = (OUT / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import requests
    from huggingface_hub import HfApi, configure_http_backend, hf_hub_download
    def backend():
        session = requests.Session(); session.trust_env = False; return session
    configure_http_backend(backend_factory=backend)
    info = HfApi(endpoint="https://hf-mirror.com", token=False).dataset_info(REPO, revision=REV, files_metadata=True)
    files = [f for suite in SUITES for f in sorted(info.siblings, key=lambda x: x.rfilename)
             if f.rfilename.startswith(suite + "/") and f.rfilename.endswith(".hdf5")]
    if len(files) != 40: raise RuntimeError(f"Expected 40 demonstration files, got {len(files)}")
    state = dict(pid=os.getpid(), source=REPO, revision=REV, endpoint="https://hf-mirror.com",
                 proxy_disabled=True, phase="downloading", planned_files=40,
                 planned_bytes=sum(f.size or 0 for f in files), files=[])
    def save():
        state["updated"] = time.strftime("%FT%T%z")
        tmp = OUT / "status.tmp"; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(OUT / "status.json")
    save()
    for file in files:
        row = dict(name=file.rfilename, bytes=file.size, status="downloading")
        state["files"].append(row); save()
        for attempt in range(1, 4):
            try:
                path = Path(hf_hub_download(REPO, file.rfilename, repo_type="dataset", revision=REV,
                                            local_dir=str(OUT), endpoint="https://hf-mirror.com", token=False))
                if path.stat().st_size != file.size: raise RuntimeError("File size mismatch")
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
                row.update(status="downloaded", sha256=h.hexdigest())
                print("DONE",file.rfilename,flush=True); break
            except Exception as exc:
                row.update(attempt=attempt, error_type=type(exc).__name__)
                if attempt == 3: row["status"] = "failed"
                else: time.sleep(15)
            finally: save()
    state["phase"] = "downloaded" if all(f["status"] == "downloaded" for f in state["files"]) else "partial_failure"
    save()


if __name__ == "__main__": main()
