import argparse
import json
import sys
from pathlib import Path

from .storage import ROOT
from .workflow import execute, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="材料手册工作台：网页操作与 Codex 模型处理"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="action", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8766)
    for action in ("intake", "figures", "tables", "apply-luna", "assemble", "status"):
        command = commands.add_parser(action)
        command.add_argument("manual")
        if action == "figures":
            command.add_argument("--figure", action="append", dest="ids")
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--retry", action="store_true")
        if action == "assemble":
            command.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.action == "serve":
        from .server import serve

        serve(args.root, args.port)
        return 0
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"root", "action", "manual"}
    }
    try:
        result = (
            summary(args.root, args.manual)
            if args.action == "status"
            else execute(args.root, args.manual, args.action, **options)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
