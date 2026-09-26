import argparse
import json
from pathlib import Path

from .pipeline import analyze
from .viewer import build_viewer
from .evaluate import run_eval


def main():
    ap = argparse.ArgumentParser(prog="bullseye", description="Explain and risk-rate functions in legacy embedded C.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="repo -> data JSON")
    a.add_argument("root")
    a.add_argument("--platform", required=True, help="name in bullseye/platforms or a path to a JSON config")
    a.add_argument("--label")
    a.add_argument("--git", action="store_true", help="mine commit messages for the top-ranked functions")
    a.add_argument("--git-top", type=int, default=80)
    a.add_argument("--llm-top", type=int, default=60, help="how many of the riskiest functions to send to the LLM (0 = none)")
    a.add_argument("-o", "--out", required=True)
    v = sub.add_parser("viewer", help="data JSON(s) -> one self-contained HTML file")
    v.add_argument("data", nargs="+")
    v.add_argument("-o", "--out", required=True)
    sub.add_parser("llm-check", help="test the LLM connection with one tiny request")
    e = sub.add_parser("eval", help="compare scores with hand-labelled expectations")
    e.add_argument("data")
    e.add_argument("expectations")
    args = ap.parse_args()

    if args.cmd == "analyze":
        data = analyze(args.root, args.platform, use_git=args.git, git_top=args.git_top, llm_top=args.llm_top, label=args.label)
        Path(args.out).write_text(json.dumps(data))
        print(f"wrote {args.out}: {len(data['functions'])} functions, {data['meta']['counts']}")
    elif args.cmd == "viewer":
        build_viewer([json.loads(Path(p).read_text()) for p in args.data], args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "llm-check":
        from .llm import check
        raise SystemExit(check())
    elif args.cmd == "eval":
        raise SystemExit(run_eval(args.data, args.expectations))
