"""Re-render training-demo states at evaluation resolution, without labels."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "calib/libero_training_demos"


def relocate_xml(xml):
    import robosuite
    from libero.libero import get_libero_path
    tree = ET.fromstring(xml)
    for node in tree.iter():
        file = node.get("file")
        if not file: continue
        parts = Path(file).parts
        if "robosuite" in parts:
            idx = max(i for i,p in enumerate(parts) if p == "robosuite")
            target = Path(robosuite.__file__).parent.joinpath(*parts[idx+1:])
        elif "assets" in parts:
            # Official demos retain the pre-release package name chiliocosm
            # and author-machine absolute paths. Resolve only local assets.
            idx = max(i for i,p in enumerate(parts) if p == "assets")
            asset_root = Path(get_libero_path("assets")).resolve()
            target = asset_root.joinpath(*parts[idx+1:]).resolve()
            if not target.is_relative_to(asset_root): raise ValueError("XML asset escapes asset root")
        else: target = Path(file)
        if not target.is_file(): raise FileNotFoundError(f"Missing local XML asset: {target}")
        node.set("file", str(target))
    return ET.tostring(tree, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "manifest.json").exists(): raise RuntimeError("Use a new output directory")
    import h5py
    import numpy as np
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task)
    relative = args.suite + "/" + task.name + "_demo.hdf5"
    downloaded = json.loads((DATA / "status.json").read_text())
    row = next((r for r in downloaded["files"] if r["name"] == relative and r["status"] == "downloaded"), None)
    if row is None: raise RuntimeError(f"Training file not downloaded and verified: {relative}")
    source = DATA / relative
    args.out.mkdir(parents=True, exist_ok=True)
    keys = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    manifest = dict(source="official_training_demonstrations", repo=downloaded["source"],
                    revision=downloaded["revision"], source_path=str(source), source_sha256=row["sha256"],
                    suite=args.suite, task_id=args.task, instruction=task.language,
                    mujoco=mujoco.__version__, resolution=256, seed=42, observations=[])
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256)
    try:
        with h5py.File(source, "r") as f:
            for demo_id in range(10):
                demo = f[f"data/demo_{demo_id}"]
                split = "calibration" if demo_id < 8 else "development"
                xml = demo.attrs["model_file"]
                if isinstance(xml, bytes): xml = xml.decode()
                env.seed(42); env.reset(); env.reset_from_xml_string(relocate_xml(xml))
                frames = np.linspace(0, len(demo["states"]) - 1, 4, dtype=int)
                for frame in frames:
                    state = np.asarray(demo["states"][int(frame)])
                    if len(state) != len(env.get_sim_state()): raise ValueError("Demo/simulator state layout mismatch")
                    obs = env.set_init_state(state)
                    payload = {k: np.asarray(obs[k]) for k in keys}
                    if any(not np.isfinite(v).all() for v in payload.values()): raise ValueError("Non-finite observation")
                    if any(payload[k].shape != (256,256,3) for k in keys[:2]): raise ValueError("Camera shape mismatch")
                    name = f"demo{demo_id:02d}_frame{frame:04d}.npz"
                    path = args.out / name
                    np.savez_compressed(path, **payload)
                    manifest["observations"].append(dict(file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                        split=split, demo_id=demo_id, frame_id=int(frame),
                                                        state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                                                        xml_sha256=hashlib.sha256(xml.encode()).hexdigest()))
    finally:
        env.close()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"observations":len(manifest["observations"]),"calibration":32,"development":8,"out":str(args.out)}))


if __name__ == "__main__": main()
