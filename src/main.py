"""                                                                                                                                                                                                                                    
main.py                                                                                                                                                                                                                                
                                                                                                                                                                                                                                       
This module serves as the entry point for the Enriched Sentence Analysis Pipeline.                                                                                                                                                     
It handles command-line arguments for input and output directories and initiates                                                                                                                                                       
the pipeline processing.                                                                                                                                                                                                               
""" 
import argparse
import asyncio
from pathlib import Path
from src.pipeline import run_pipeline
from src.config import config
from src.utils.logger import get_logger

logger = get_logger()

def main():
    """"
     Main function to execute the Enriched Sentence Analysis Pipeline.                                                                                                                                                                  
                                                                                                                                                                                                                                       
    This function sets up the argument parser for input and output directories,                                                                                                                                                        
    logs the start of the pipeline execution, and runs the pipeline asynchronously.                                                                                                                                                    
                                                                                                                                                                                                                                       
    Returns: 
        None
    """
    # Create an argument parser to handle command-line arguments 
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
    # Parse the command-line arguments
    args = parser.parse_args()

    # Log the start of the pipeline execution 
    logger.info("Starting the Enriched Sentence Analysis Pipeline")
    asyncio.run(run_pipeline(args.input_dir, args.output_dir))
    # Log the successful completion of the pipeline execution 
    logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    main()
