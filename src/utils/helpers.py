"""
helpers.py

Provides utility functions for common data handling tasks, including loading and
saving data in JSON, YAML, and pandas DataFrame formats (CSV/Excel).
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import yaml


def save_json(data: Any, file_path: Union[str, Path], indent: int = 4):
    """
    Save Python data structure to a JSON file.

    Ensures the target directory exists before writing.

    Args:
        data (Any): The Python object (e.g., dict, list) to serialize.
        file_path (Union[str, Path]): The path to the output JSON file.
        indent (int): The indentation level for pretty-printing the JSON.
                      Defaults to 4.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.

    Args:
        file_path (Union[str, Path]): The path to the input JSON file.

    Returns:
        Any: The Python object deserialized from the JSON file.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_json_line(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Append a dictionary as a JSON line (JSONL format) to a file.

    Ensures the directory exists and appends the JSON string followed by a newline.
    Useful for logging structured data or creating JSON Lines files.

    Args:
        data (Dict[str, Any]): The dictionary data to append.
        file_path (Union[str, Path]): The path to the file.
    """
    file_path = Path(file_path)
    # Ensure the output directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert the dictionary to a JSON string
    json_string = json.dumps(data, ensure_ascii=False)

    # Append the JSON string followed by a newline
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json_string + "\n")


def save_yaml(data: dict, file_path: Union[str, Path]):
    """
    Save a dictionary to a YAML file.

    Ensures the target directory exists before writing.

    Args:
        data (dict): The dictionary to serialize.
        file_path (Union[str, Path]): The path to the output YAML file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_yaml(file_path: Union[str, Path]) -> dict:
    """
    Load data from a YAML file.

    Args:
        file_path (Union[str, Path]): The path to the input YAML file.

    Returns:
        dict: The dictionary deserialized from the YAML file.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        yaml.YAMLError: If the file content is not valid YAML.
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataframe(
    df: pd.DataFrame, file_path: Union[str, Path], file_type: str = "csv"
):
    """
    Save a pandas DataFrame to a CSV or Excel file.

    Ensures the target directory exists before writing.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (Union[str, Path]): The path for the output file.
        file_type (str): The type of file to save ('csv' or 'excel').
                         Defaults to 'csv'.

    Raises:
        ValueError: If file_type is not 'csv' or 'excel'.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_type.lower() == "csv":
        df.to_csv(file_path, index=False)
    elif file_type.lower() == "excel":
        df.to_excel(file_path, index=False)
    else:
        raise ValueError("Unsupported file_type. Choose 'csv' or 'excel'.")


def load_dataframe(file_path: Union[str, Path], file_type: str = "csv") -> pd.DataFrame:
    """
    Load a pandas DataFrame from a CSV or Excel file.

    Args:
        file_path (Union[str, Path]): The path to the input file.
        file_type (str): The type of file to load ('csv' or 'excel').
                         Defaults to 'csv'.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If file_type is not 'csv' or 'excel'.
        Exception: Other exceptions specific to pandas readers (e.g., parsing errors).
    """
    file_path = Path(file_path)
    if file_type.lower() == "csv":
        return pd.read_csv(file_path)
    elif file_type.lower() == "excel":
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file_type. Choose 'csv' or 'excel'.")
