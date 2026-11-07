#!/usr/bin/env python3
"""One entry point for the whole project: Task 1 -> Task 2 -> Task 3 -> REPORT.md.

  python run_all.py                       # everything, config.yaml
  python run_all.py --tasks 2 3           # subset (Task 3 needs Task 1 + Task 2 outputs)
  python run_all.py --config config.smoke.yaml
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from p5 import report, task1_ppmi, task2_probe, task3_bert_static  # noqa: E402
from p5.utils import LOG, load_config, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    args = ap.parse_args()
    overrides = {"device": args.device} if args.device else {}
    cfg = load_config(args.config, overrides)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "run.log")
    LOG.info("Config: %s | tasks %s", Path(args.config).resolve(), args.tasks)
    steps = {1: task1_ppmi.run, 2: task2_probe.run, 3: task3_bert_static.run}
    for t in sorted(args.tasks):
        t0 = time.time()
        LOG.info("=" * 25 + f" TASK {t} " + "=" * 25)
        steps[t](cfg)
        LOG.info("Task %d finished in %.1f min", t, (time.time() - t0) / 60)
    report.build(cfg)
    LOG.info("All done. Results in %s", cfg["paths"]["output_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
