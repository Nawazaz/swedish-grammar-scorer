"""Run the whole pipeline on your own machine: check setup, build data, train,
evaluate and (optionally) publish. Settings adapt to GPU or CPU.

    python run_local.py --quick                 # ~5-10 min test run, small subset
    python run_local.py                         # real run
    python run_local.py --hf-user <name>        # real run + publish data and model

Every step is a normal script, so you can also run them one by one
(see README). On Windows, run this with `python`, not a venv .exe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title: str, args: list[str]) -> None:
    print(f"\n=== {title} ===\n$ python {' '.join(args)}", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"--- {title} done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


def check_setup() -> bool:
    """Returns True if a CUDA GPU is available. Exits with advice if torch is broken."""
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        sys.exit(f"Could not import a dependency: {e}\n"
                 "Install them with:  python -m pip install -r requirements.txt\n"
                 "If torch is installed but blocked (e.g. Windows Application Control), use the Colab notebook.")
    gpu = torch.cuda.is_available()
    print(f"torch {torch.__version__} | device: {torch.cuda.get_device_name(0) if gpu else 'CPU only'}")
    return gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small end-to-end test run")
    parser.add_argument("--hf-user", help="publish to <hf-user>/swedish-grammar-data and <hf-user>/swedish-grammar-scorer")
    parser.add_argument("--data", default="data/processed", help="dataset folder (built if missing)")
    parser.add_argument("--rebuild-data", action="store_true", help="rebuild the dataset even if it exists")
    parser.add_argument("--model-dir", default="models/scorer")
    parser.add_argument("--base-model", help="override the encoder to fine-tune")
    args = parser.parse_args()

    gpu = check_setup()

    # GPU: full data. CPU: a subset, so training finishes in about an hour.
    if args.quick:
        wiki, train_samples, epochs = 2_000, 2_000, 1
    elif gpu:
        wiki, train_samples, epochs = 30_000, None, 2
    else:
        wiki, train_samples, epochs = 30_000, 15_000, 1
        print("No GPU found: training on a 15k-example subset for 1 epoch (about 1 hour on a laptop CPU).")

    if args.hf_user and (args.quick or train_samples):
        print("Note: publishing a quick or CPU-subset run. Mention these settings in the paper.")

    data_ready = (ROOT / args.data / "train.jsonl").exists()
    if args.rebuild_data or not data_ready:
        cmd = ["build_dataset.py", "--out", args.data, "--wiki-passages", str(wiki)]
        if args.hf_user:
            cmd += ["--push", f"{args.hf_user}/swedish-grammar-data"]
        step("1/3 Build dataset", cmd)
    else:
        print(f"\nUsing existing dataset in {args.data} (pass --rebuild-data to rebuild).")

    cmd = ["train.py", "--data", args.data, "--output-dir", args.model_dir, "--epochs", str(epochs)]
    if train_samples:
        cmd += ["--max-train-samples", str(train_samples)]
    if args.base_model:
        cmd += ["--base-model", args.base_model]
    if args.hf_user:
        cmd += ["--push", f"{args.hf_user}/swedish-grammar-scorer"]
    step("2/3 Train", cmd)

    step("3/3 Evaluate on starter test set", ["evaluate.py", "--model", args.model_dir])

    if args.hf_user:
        print(f"\nDone. Set MODEL_ID = \"{args.hf_user}/swedish-grammar-scorer\" in predict.py before you submit.")


if __name__ == "__main__":
    main()
