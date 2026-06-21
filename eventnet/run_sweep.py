"""Train + paper-eval a set of (feature_mode, K) configs, fanned over GPUs.

Each job runs ``train`` then ``evaluate`` (peak-level F1) on one GPU; up to one
job per GPU runs concurrently. Default plan = the ablation table at K=4 (all 6
feature modes) + the K-sweep {1,2,4,8} for the proposed ``tdtaw``.

Example:
  PYTHONPATH=<repo>/src uv run python -m eventnet.run_sweep \
      --save_root outputs/eventnet/sweep --epochs 40 --gpus 0 1 2 3
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envconfig  # noqa: E402  (machine-dependent paths; see env.yaml.example)

DEFAULT_JOBS = [
    ("t_only", 4), ("t_dt", 4), ("ta", 4), ("tdta", 4), ("taw", 4), ("tdtaw", 4),
    ("tdtaw", 1), ("tdtaw", 2), ("tdtaw", 8),
]


def job_cmd(mode, k, gpu, save_root, epochs, frame_stride, eval_stride, extra):
    sd = os.path.join(save_root, f"{mode}_K{k}")
    dev = f"cuda:{gpu}"
    train = (f"python -m eventnet.train --K {k} --feature_mode {mode} "
             f"--frame_stride {frame_stride} --epochs {epochs} --device {dev} "
             f"--save_dir {sd} {extra}")
    ev = (f"python -m eventnet.evaluate --checkpoint {sd}/best.pth "
          f"--frame_stride {eval_stride} --device {dev} --out {sd}/eval.json")
    return sd, f"set -e; {train}; {ev}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_root", default=envconfig.output_path("eventnet", "sweep"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--frame_stride", type=int, default=7)
    ap.add_argument("--eval_stride", type=int, default=3)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--jobs", nargs="+", default=None,
                    help='override plan, e.g. tdtaw:4 tdta:4')
    ap.add_argument("--train_extra", default="")
    args = ap.parse_args()

    jobs = DEFAULT_JOBS
    if args.jobs:
        jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in args.jobs]
    os.makedirs(args.save_root, exist_ok=True)

    pending = list(jobs)
    running = {}  # gpu -> (proc, save_dir, logf)
    free = list(args.gpus)
    print(f"[sweep] {len(jobs)} jobs over GPUs {args.gpus}")

    def launch(mode, k, gpu):
        sd, cmd = job_cmd(mode, k, gpu, args.save_root, args.epochs,
                          args.frame_stride, args.eval_stride, args.train_extra)
        logf = open(os.path.join(args.save_root, f"{mode}_K{k}.log"), "w")
        p = subprocess.Popen(["bash", "-c", cmd], stdout=logf, stderr=subprocess.STDOUT)
        print(f"  launch {mode} K{k} on cuda:{gpu} (pid {p.pid}) -> {sd}")
        return p, sd, logf

    while pending or running:
        while pending and free:
            mode, k = pending.pop(0)
            gpu = free.pop(0)
            running[gpu] = launch(mode, k, gpu)
        time.sleep(10)
        for gpu, (p, sd, logf) in list(running.items()):
            if p.poll() is not None:
                logf.close()
                ok = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"  done cuda:{gpu} {sd} [{ok}]")
                del running[gpu]
                free.append(gpu)
    print("[sweep] all jobs finished")


if __name__ == "__main__":
    main()
