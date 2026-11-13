"""CLI: export / export-initial / validate / describe an M1.5 bundle.

    # before training: deterministic step-0 baseline
    python -m autotrain.export export-initial \
        --config autotrain/configs/wanxiang-smoke-4m.yaml --out runs/smoke-4m/bundle_step0

    # after training: export the last checkpoint
    python -m autotrain.export export --checkpoint runs/x/checkpoints/ckpt_step_500.pt \
        --config autotrain/configs/wanxiang-smoke-4m.yaml --out runs/x/bundle
    python -m autotrain.export validate --bundle runs/x/bundle --checkpoint runs/x/checkpoints/ckpt_step_500.pt
    python -m autotrain.export describe --bundle runs/x/bundle
"""

from __future__ import annotations

import argparse
import json

from .contract import validate_bundle_files
from .exporter import export_checkpoint, export_initial
from .loader import validate_bundle


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wanxiang M1.5 model bundle")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="checkpoint -> bundle")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--config", required=True)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--force", action="store_true")

    p_initial = sub.add_parser("export-initial", help="untrained step-0 baseline bundle")
    p_initial.add_argument("--config", required=True)
    p_initial.add_argument("--out", required=True)
    p_initial.add_argument("--seed", type=int, default=None)
    p_initial.add_argument("--force", action="store_true")

    p_validate = sub.add_parser("validate", help="bundle vs source checkpoint")
    p_validate.add_argument("--bundle", required=True)
    p_validate.add_argument("--checkpoint", required=True)

    p_describe = sub.add_parser("describe", help="print bundle metadata")
    p_describe.add_argument("--bundle", required=True)

    args = parser.parse_args()
    if args.command == "export":
        _print(export_checkpoint(args.checkpoint, args.config, args.out, force=args.force))
    elif args.command == "export-initial":
        _print(
            export_initial(
                args.config, args.out, seed=args.seed, force=args.force
            )
        )
    elif args.command == "validate":
        _print(validate_bundle(args.bundle, args.checkpoint))
    else:
        config, tokenizer, manifest = validate_bundle_files(args.bundle)
        _print({"config": config, "tokenizer": tokenizer, "manifest": manifest})


if __name__ == "__main__":
    main()
