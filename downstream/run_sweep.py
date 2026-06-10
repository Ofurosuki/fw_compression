"""Run the downstream compression sweep across GPUs (evaluator A = FWL-ToPM).

Each job evaluates one (method, K) on the split2 test set (subsampled by --divide
for speed) and writes <tag>.json + a 6-waveform viz <tag>.png. Jobs are distributed
round-robin over the given GPUs and run concurrently (one job per GPU at a time).
"""
import os
import subprocess
import threading

REPO = "/data3/user/yoshida/fwl_mae/neurips2026"
CFG = "downstream/configs/evalA_split2_test.yaml"
DIVIDE = "3"
OUT = "downstream/outputs/sweep"
VIZ = OUT + "/viz"
GPUS = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
os.makedirs(VIZ, exist_ok=True)


def ck1d(enc, K):
    return f"runs/real_split2_1d/{enc}_K{K}/checkpoint.pt"


def cksp(K):
    return f"runs/real_split2_spatial/spatial_K{K}/checkpoint.pt"


jobs = [("none", ["--compress", "none"])]
for K in [8, 16, 32, 64, 128]:
    jobs.append((f"1d_ll_K{K}", ["--compress", "ae", "--ae_ckpt", ck1d("learnable_linear", K)]))
for K in [128, 256, 512, 1024, 2048]:
    jobs.append((f"sp_K{K}", ["--compress", "ae", "--ae_ckpt", cksp(K)]))
for K in [8, 32, 128]:
    jobs.append((f"1d_cb_K{K}", ["--compress", "ae", "--ae_ckpt", ck1d("coarse_binning", K)]))

queues = {g: [] for g in GPUS}
for i, job in enumerate(jobs):
    queues[GPUS[i % len(GPUS)]].append(job)

env = dict(os.environ, PYTHONPATH=f"{REPO}/src", PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])


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
print("SWEEP COMPLETE", flush=True)
