"""CLI: make-corpus / verify / run / compare.

    python -m autotrain.eval make-corpus --out runs/train_tokens.bin
    python -m autotrain.eval verify
    python -m autotrain.eval run --bundle runs/x/bundle --out runs/report.json
    python -m autotrain.eval compare runs/before.json runs/after.json

``make-corpus`` is the only subcommand that writes tokens. It writes to the
path you give it, never into ``evals/`` (the frozen corpora stay read-only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evalspec import (
    CORPUS_DIR,
    CORPUS_TOKENS,
    MARKOV_P,
    MIN_IMPROVEMENT_BITS,
    SEED_TRAIN_CORPUS,
    VOCAB_SIZE,
    verify_corpus,
)


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def _write_report(obj: dict, out: str | None) -> None:
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    _print(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wanxiang M1.5 evaluator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make-corpus", help="generate structured training tokens")
    p_make.add_argument("--out", required=True)
    p_make.add_argument("--tokens", type=int, default=2_000_000)

    p_verify = sub.add_parser("verify", help="verify frozen eval corpora")
    p_verify.add_argument("--eval-dir", default=None)

    p_run = sub.add_parser("run", help="evaluate a bundle")
    p_run.add_argument("--bundle", required=True)
    p_run.add_argument("--out", default=None)
    p_run.add_argument("--eval-dir", default=None)
    p_run.add_argument("--device", default=None)

    p_cmp = sub.add_parser("compare", help="A/B two reports")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")
    p_cmp.add_argument("--out", default=None)

    args = parser.parse_args()

    if args.command == "make-corpus":
        from .corpus import make_markov_tokens, sha256_file, write_tokens

        tokens = make_markov_tokens(
            args.tokens, VOCAB_SIZE, MARKOV_P, SEED_TRAIN_CORPUS
        )
        write_tokens(args.out, tokens)
        _print(
            {
                "path": args.out,
                "tokens": int(len(tokens)),
                "vocab_size": VOCAB_SIZE,
                "markov_p": MARKOV_P,
                "seed": SEED_TRAIN_CORPUS,
                "sha256": sha256_file(args.out),
                "note": "Same distribution as the frozen eval corpus, disjoint seed.",
            }
        )
    elif args.command == "verify":
        manifest = verify_corpus(args.eval_dir)
        _print(
            {
                "status": "OK",
                "eval_dir": str(Path(args.eval_dir) if args.eval_dir else CORPUS_DIR),
                "manifest": manifest,
            }
        )
    elif args.command == "run":
        from .runner import EvalRunner

        report = EvalRunner(
            args.bundle, eval_dir=args.eval_dir, device=args.device
        ).run()
        _write_report(report, args.out)
    else:
        from .runner import compare_reports

        with open(args.before, "r", encoding="utf-8") as f:
            before = json.load(f)
        with open(args.after, "r", encoding="utf-8") as f:
            after = json.load(f)
        result = compare_reports(before, after)
        result["min_improvement_bits"] = MIN_IMPROVEMENT_BITS
        _write_report(result, args.out)


if __name__ == "__main__":
    main()
