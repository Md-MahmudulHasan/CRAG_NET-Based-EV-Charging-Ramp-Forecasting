"""Simple file-locked job queue so several worker processes can share an experiment list.

python src/runner.py QUEUE.json [--max_minutes M]
Each job: {"dataset":..., "model":..., "seed":..., "tag":..., "cfg":{...}, "epochs":..., "max_batches":...}
A job is skipped if its metrics.json exists or another worker holds its lock directory.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(qfile, max_minutes=110):
    t0 = time.time()
    while True:
        jobs = json.load(open(qfile))
        todo = None
        for j in jobs:
            rd = "runs_torch" if j.get("backend", "torch") == "torch" else "runs"
            out = os.path.join(ROOT, "results", rd, j["dataset"], f"{j['tag']}_s{j['seed']}")
            if os.path.exists(os.path.join(out, "metrics.json")):
                continue
            lock = os.path.join(ROOT, "results", "locks", f"{j['dataset']}__{j['tag']}__{j['seed']}")
            os.makedirs(os.path.dirname(lock), exist_ok=True)
            try:
                os.mkdir(lock)
            except FileExistsError:
                pidf = os.path.join(lock, "pid")
                try:
                    pid = int(open(pidf).read())
                    os.kill(pid, 0)              # alive -> another worker owns it
                    continue
                except (OSError, ValueError, FileNotFoundError):
                    pass                         # stale lock from a killed worker: take it over
            with open(os.path.join(lock, "pid"), "w") as f:
                f.write(str(os.getpid()))
            todo = (j, lock)
            break
        if todo is None:
            print("queue empty", flush=True)
            return
        j, lock = todo
        if (time.time() - t0) / 60 > max_minutes:
            import shutil
            shutil.rmtree(lock, ignore_errors=True)
            print("time budget reached; exiting so the job can be resumed", flush=True)
            return
        script = "train_torch.py" if j.get("backend", "torch") == "torch" else "train.py"
        cmd = [sys.executable, "-I", os.path.join(ROOT, "src", script), "--dataset", j["dataset"],
               "--model", j["model"], "--seed", str(j["seed"]), "--tag", j["tag"],
               "--epochs", str(j.get("epochs", 25)), "--max_batches", str(j.get("max_batches", 30)),
               "--patience", str(j.get("patience", 8)), "--cfg", json.dumps(j.get("cfg", {})),
               "--lr", str(j.get("lr", 2e-3))]
        if script == "train.py":
            cmd += ["--sched", j.get("sched", "cosine")]
        log = os.path.join(ROOT, "logs", f"{j['dataset']}__{j['tag']}__s{j['seed']}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        print(time.strftime("%H:%M:%S"), "start", j["dataset"], j["tag"], j["seed"], flush=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(log, "w") as f:
            r = subprocess.run(cmd + ["--verbose"], stdout=f, stderr=subprocess.STDOUT, env=env)
        import shutil
        shutil.rmtree(lock, ignore_errors=True)
        tail = open(log).read().strip().splitlines()[-1:] if os.path.exists(log) else ["?"]
        print(time.strftime("%H:%M:%S"), "end", r.returncode, tail[0][:200], flush=True)


if __name__ == "__main__":
    mm = 110
    if "--max_minutes" in sys.argv:
        mm = float(sys.argv[sys.argv.index("--max_minutes") + 1])
    main(sys.argv[1], mm)
