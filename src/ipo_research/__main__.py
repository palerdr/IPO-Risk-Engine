"""Run the IPO research commands from the repository root."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="IPO drawdown research")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Freeze the public price sample")
    fetch.add_argument("--universe", type=Path, default=Path("research/universe.json"))
    fetch.add_argument("--output", type=Path, default=Path("research/input.json"))
    fetch.add_argument("--refresh", action="store_true")
    run = commands.add_parser("evaluate", help="Run chronological evaluation from a frozen input")
    run.add_argument("--input", type=Path, default=Path("research/input.json"))
    run.add_argument("--output", type=Path, default=Path("web/data/research.json"))
    args = parser.parse_args()
    if args.command == "fetch":
        from .data import fetch_cohort
        result = fetch_cohort(args.universe, args.output, args.refresh)
        print(f"Fetched {len(result['listings'])}; excluded {len(result['exclusions'])}. Snapshot: {args.output}")
    else:
        from .evaluate import run_evaluation
        result = run_evaluation(args.input, args.output)
        print(f"Evaluated {len(result['predictions'])} held-out listings. Report: {args.output}")


if __name__ == "__main__":
    main()
