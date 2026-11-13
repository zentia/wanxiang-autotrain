"""CLI entry: python -m autotrain.trainer --config configs/wanxiang-100m.yaml --data tokens.bin --out runs/wanxiang-100m"""

from __future__ import annotations

import argparse
import logging

from ..config import load_config
from ..store.experiment_store import ExperimentStore
from .runner import TrainerRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Wanxiang Trainer Runner (M1)")
    parser.add_argument("--config", required=True, help="experiment YAML config")
    parser.add_argument("--data", required=True, help="uint16 token file")
    parser.add_argument("--out", required=True, help="output run directory")
    parser.add_argument("--db", default=None, help="experiment store sqlite path (optional)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    store = ExperimentStore(args.db) if args.db else None
    runner = TrainerRunner(cfg, args.data, args.out, store=store)
    summary = runner.train()
    print(summary)


if __name__ == "__main__":
    main()
