"""Run the expanded research pipeline from the repository root."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Expanded pre-IPO and day-20 challenger study")
    parser.add_argument(
        "command", choices=("registry", "fetch", "build", "train", "report", "verify", "audit")
    )
    parser.add_argument("--directory", type=Path, default=Path("research/expanded"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/challengers"))
    parser.add_argument(
        "--report-directory",
        type=Path,
        default=Path("docs"),
        help="Destination for generated Markdown (report command)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if args.command == "registry":
        from .sources import registry

        registry(args.directory)
    elif args.command == "fetch":
        from .sources import fetch_prices

        fetch_prices(args.directory, workers=args.workers, retry_failures=args.retry_failures)
    elif args.command == "build":
        from .dataset import build

        build(args.directory)
    elif args.command == "train":
        from .evaluate import train

        train(args.directory, args.artifacts)
        train(args.directory, args.artifacts, sensitivity=True)
    elif args.command == "report":
        from .report import report

        report(args.directory, args.report_directory)
    elif args.command == "audit":
        from .decision_audit import audit

        output = args.directory.parent / "decision/audit.json"
        audit(args.directory, output)
        print(f"Decision audit: {output}")
    else:
        from .verify import verify

        verify(args.directory, args.artifacts)
        verify(args.directory, args.artifacts, sensitivity=True)


if __name__ == "__main__":
    main()
