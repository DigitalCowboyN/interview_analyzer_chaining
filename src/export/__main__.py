"""Export an interview x lens as an OKF bundle.

Usage: python -m src.export <interview_id> <lens_name> [--out exports] [--zip]
"""

import argparse
import asyncio
import sys

from src.export.bundler import OkfExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an OKF bundle (Layer 5)")
    parser.add_argument("interview_id")
    parser.add_argument("lens_name")
    parser.add_argument("--out", default="exports", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Also produce a .zip archive")
    args = parser.parse_args()

    try:
        result = asyncio.run(
            OkfExporter().export(
                args.interview_id, args.lens_name, out_dir=args.out, zip_bundle=args.zip
            )
        )
    except (ValueError, RuntimeError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
