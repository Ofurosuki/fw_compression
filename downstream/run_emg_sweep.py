"""Gaussian vs EMG synthesis kernel, taw, K in {2,3,4}, divide=3, neurips_best.

Tests whether an asymmetric (right-tailed) EMG synthesis kernel — which matches
real FW-LiDAR returns better than a symmetric Gaussian (measured skew ≈ +0.78,
see analyze_pulse_shape.py) — improves frozen Ghost-FWL downstream F1.
"""
import os
import subprocess
import threading

REPO = "/data3/user/yoshida/fwl_mae/neurips2026"
CFG = "downstream/configs/evalA_split2_test_best.yaml"
DIVIDE = "3"
OUT = "downstream/outputs/emg"
VIZ = OUT + "/viz"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
os.makedirs(VIZ, exist_ok=True)

jobs = []
for kernel in ["gaussian", "emg"]:
    for K in [2, 3, 4]:
        tag = f"taw_K{K}_{kernel}"
        jobs.append((tag, ["--compress", "event", "--event_k", str(K),
                           "--event_repr", "taw", "--event_kernel", kernel]))

queues = {g: [] for g in GPUS}
for i, job in enumerate(jobs):
    queues[GPUS[i % len(GPUS)]].append(job)

env = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])


def worker(gpu, joblist):
    for tag, jargs in joblist:
        cmd = ["uv", "run", "python", "downstream/run_eval.py", "--config", CFG,
               "--device", gpu, "--divide", DIVIDE, "--out", f"{OUT}/{tag}.json",
               "--viz_out", f"{VIZ}/{tag}.png"] + jargs
        print(f"START {gpu} {tag}", flush=True)
        with open(f"{OUT}/{tag}.log", "w") as f:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"DONE  {gpu} {tag} rc={r.returncode}", flush=True)


ts = [threading.Thread(target=worker, args=(g, queues[g])) for g in GPUS]
for t in ts:
    t.start()
for t in ts:
    t.join()
print("EMG_SWEEP COMPLETE", flush=True)
