"""Downstream sweep for the top-K transport-EVENT representation.

For each (K, representation) we extract top-K events from every raw waveform,
synthesise a Gaussian-pulse pseudo-waveform, and run the frozen FWL-ToPM model —
the same harness as run_sweep.py, but with --compress event instead of an AE.

K          ∈ {1,2,3,4,6,8}
repr       ∈ {t, ta, tw, taw}   (position / +intensity / +width / all three)
divide=3 for speed. Writes events_K{K}_{repr}.json + a 6-waveform viz.
"""
import argparse
import os
import subprocess
import threading

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="downstream/configs/evalA_split2_test.yaml")
ap.add_argument("--out", default="downstream/outputs/events")
ap.add_argument("--divide", default="3")
cli = ap.parse_args()

REPO = "/data3/user/yoshida/fwl_mae/neurips2026"
CFG = cli.config
DIVIDE = cli.divide
OUT = cli.out
VIZ = OUT + "/viz"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
KS = [1, 2, 3, 4, 6, 8]
REPRS = ["t", "ta", "tw", "taw"]
os.makedirs(VIZ, exist_ok=True)

jobs = [("none", ["--compress", "none"])]
for repr_ in REPRS:
    for K in KS:
        tag = f"events_K{K}_{repr_}"
        jobs.append((tag, ["--compress", "event", "--event_k", str(K),
                           "--event_repr", repr_]))

queues = {g: [] for g in GPUS}
for i, job in enumerate(jobs):
    queues[GPUS[i % len(GPUS)]].append(job)

env = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])


def worker(gpu, joblist):
    for tag, jargs in joblist:
        cmd = ["uv", "run", "python", "downstream/run_eval.py", "--config", CFG,
               "--device", gpu, "--divide", DIVIDE, "--out", f"{OUT}/{tag}.json"] + jargs
        if "event" in jargs:
            cmd += ["--viz_out", f"{VIZ}/{tag}.png"]
        print(f"START {gpu} {tag}", flush=True)
        with open(f"{OUT}/{tag}.log", "w") as f:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"DONE  {gpu} {tag} rc={r.returncode}", flush=True)


ts = [threading.Thread(target=worker, args=(g, queues[g])) for g in GPUS]
for t in ts:
    t.start()
for t in ts:
    t.join()
print("SWEEP_EVENTS COMPLETE", flush=True)
