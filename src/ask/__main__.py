"""CLI: python -m src.ask <project_id> "<question>" [--top-k 12]"""

import argparse
import asyncio
import sys

from src.ask.engine import AskEngine, SynthesisUnavailable


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a project a question (M4.6)")
    parser.add_argument("project_id")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    try:
        result = asyncio.run(
            AskEngine().ask(args.project_id, args.question, top_k=args.top_k)
        )
    except SynthesisUnavailable as e:
        print(e.result.model_dump_json(indent=2))
        print(f"synthesis unavailable: {e}", file=sys.stderr)
        sys.exit(1)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
