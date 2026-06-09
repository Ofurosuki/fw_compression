"""Downstream sweep for the ANTI-HALLUCINATION-trained AEs (bg=5.0 fp=0.5).

Same harness as run_sweep.py but points at the `_ah` checkpoints and writes to
downstream/outputs/sweep_ah/. Scope = the retrained methods: 1D learnable_linear
and spatial 4x4, all K.
"""
import os
import subprocess
import threading

REPO = "/data3/user/yoshida/fwl_mae/neurips2026"
CFG = "downstream/configs/evalA_split2_test.yaml"
DIVIDE = "3"
OUT = "downstream/outputs/sweep_ah"
VIZ = OUT + "/viz"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
os.makedirs(VIZ, exist_ok=True)

jobs = [("none", ["--compress", "none"])]
for K in [8, 16, 32, 64, 128]:
    jobs.append((f"1d_ll_K{K}", ["--compress", "ae", "--ae_ckpt",
                                 f"runs/real_split2_1d_ah/learnable_linear_K{K}/checkpoint.pt"]))
for K in [128, 256, 512, 1024, 2048]:
    jobs.append((f"sp_K{K}", ["--compress", "ae", "--ae_ckpt",
                              f"runs/real_split2_spatial_ah/spatial_K{K}/checkpoint.pt"]))

queues = {g: [] for g in GPUS}
for i, job in enumerate(jobs):
    queues[GPUS[i % len(GPUS)]].append(job)

env = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])


def worker(gpu, joblist):
    for tag, args in joblist:
        cmd = ["uv", "run", "python", "downstream/run_eval.py", "--config", CFG,
               "--device", gpu, "--divide", DIVIDE, "--out", f"{OUT}/{tag}.json"] + args
        if "ae" in args:
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
print("SWEEP_AH COMPLETE", flush=True)
