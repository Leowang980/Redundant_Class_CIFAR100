"""
Run 4 experiments (100, 101, 105, 110 classes) in parallel via subprocesses.
All use cuda:0. Each logs to a separate .log file.
"""
import os
import subprocess
import sys
from pathlib import Path

# Experiment config: num_classes -> log_file
EXPERIMENTS = [
    (115, "train_115.log"),
    (120, "train_120.log"),
    (125, "train_125.log"),
    (130, "train_130.log"),
    (135, "train_135.log"),
    (140, "train_140.log"),
    (145, "train_145.log"),
    (150, "train_150.log"),
]

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    procs = []
    for num_classes, log_file in EXPERIMENTS:
        log_path = SCRIPT_DIR / "logs" / log_file
        history_path = SCRIPT_DIR / "logs" / f"history_{num_classes}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_single.py"),
            "--num_classes", str(num_classes),
            "--log_file", str(log_path),
            "--device", "cuda:0",
            "--save_history", str(history_path),
            "--seed", "42",
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        p = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((num_classes, log_file, p))
        print(f"Started experiment num_classes={num_classes}, log={log_file}, pid={p.pid}")

    print("\nAll 4 experiments running in parallel. Waiting for completion...")
    for num_classes, log_file, p in procs:
        stdout, _ = p.communicate()
        log_path = SCRIPT_DIR / "logs" / log_file
        if p.returncode != 0:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== Subprocess exited with error (exit code={p.returncode}) ===\n")
                f.write("=== Full stderr/stdout output ===\n")
                f.write(stdout if stdout else "(no output)\n")
                f.write("\n")
        status = "OK" if p.returncode == 0 else f"FAILED (code={p.returncode})"
        print(f"  num_classes={num_classes} ({log_file}): {status}")

    print("\nDone. Run 'python visualize.py' to generate loss/acc plots.")


if __name__ == "__main__":
    main()
