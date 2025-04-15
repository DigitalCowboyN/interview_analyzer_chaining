# src/utils/helpers.py
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def save_json(data, file_path: str, indent: int = 4):
    """Save data as JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> dict:
    """Load data from JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_json_line(data: Dict[str, Any], file_path: Path):
    """
    Append a dictionary as a JSON line to a file.

    Ensures the directory exists and appends the JSON string followed by a newline.

    Parameters:
        data (Dict[str, Any]): The dictionary data to append.
        file_path (Path): The path to the file.
    """
    # Ensure the output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert the dictionary to a JSON string
    json_string = json.dumps(data, ensure_ascii=False)
    
    # Append the JSON string followed by a newline
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json_string + "\n")


def save_yaml(data: dict, file_path: str):
    """Save data as YAML."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_yaml(file_path: str) -> dict:
    """Load data from YAML file."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataframe(df, file_path, file_type="csv"):
    """Save DataFrame as CSV or Excel."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_type == "csv":
        df.to_csv(file_path, index=False)
    elif file_type == "excel":
        df.to_excel(file_path, index=False)
    else:
        raise ValueError("Unsupported file_type. Choose 'csv' or 'excel'.")


def load_dataframe(file_path, file_type="csv"):
    """Load DataFrame from CSV or Excel."""
    file_path = Path(file_path)
    if file_type == "csv":
        return pd.read_csv(file_path)
    elif file_type == "excel":
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file_type. Choose 'csv' or 'excel'.")
