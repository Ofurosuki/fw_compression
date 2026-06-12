"""Evaluate every ghost-fwl split2 augmentation-variant checkpoint (no compression)
to find which one is closest to the paper's reported F1-mean (~0.592).

Each config in downstream/configs/ckptvar/ points the frozen FWL-ToPM at a different
augmentation-variant checkpoint; we run the no-compression baseline at divide=3 and
report the 3-class signal F1-mean. Jobs are fanned over GPUs.
"""
import json
import os
import subprocess
import threading

REPO = "/data3/user/yoshida/fwl_mae/neurips2026"
OUT = "downstream/outputs/ckptvar"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
DIVIDE = "3"

manifest = json.load(open(f"{OUT}/manifest.json"))
queues = {g: [] for g in GPUS}
for i, m in enumerate(manifest):
    queues[GPUS[i % len(GPUS)]].append(m)

env = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])


def worker(gpu, jobs):
    for m in jobs:
        tag = m["tag"]
        cmd = ["uv", "run", "python", "downstream/run_eval.py", "--config", m["config"],
               "--compress", "none", "--divide", DIVIDE, "--device", gpu,
               "--out", f"{OUT}/{tag}.json"]
        print(f"START {gpu} {tag}", flush=True)
        with open(f"{OUT}/{tag}.log", "w") as f:
            r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"DONE  {gpu} {tag} rc={r.returncode}", flush=True)


ts = [threading.Thread(target=worker, args=(g, queues[g])) for g in GPUS]
for t in ts:
    t.start()
for t in ts:
    t.join()
print("CKPTVAR COMPLETE", flush=True)
