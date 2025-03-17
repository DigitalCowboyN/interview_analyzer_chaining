# src/main.py
import argparse
from pathlib import Path
from src.pipeline import run_pipeline
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Enriched Sentence Analysis Pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(config["paths"]["input_dir"]),
        help="Path to the input directory containing transcript files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(config["paths"]["output_dir"]),
        help="Path to the directory for saving output analysis JSON files",
    )

    args = parser.parse_args()

    logger.info("Starting the Enriched Sentence Analysis Pipeline")
    run_pipeline(args.input_dir, args.output_dir)
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
from src.pipeline import run_pipeline
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(description="Enriched Sentence Analysis Pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(config["paths"]["input_dir"]),
        help="Path to the input directory containing transcript files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(config["paths"]["output_dir"]),
        help="Path to the directory for saving output analysis JSON files",
    )

    args = parser.parse_args()

    logger.info("Starting the Enriched Sentence Analysis Pipeline")
    run_pipeline(args.input_dir, args.output_dir)
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
