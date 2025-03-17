# src/utils/helpers.py
import json
import yaml
import pandas as pd
from pathlib import Path


def save_json(data, file_path, indent=4):
    """Save data as JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path):
    """Load data from JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data, file_path):
    """Save data as YAML."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_yaml(file_path):
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
import json
import yaml
import pandas as pd
from pathlib import Path


def save_json(data, file_path, indent=4):
    """Save data as JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path):
    """Load data from JSON file."""
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data, file_path):
    """Save data as YAML."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_yaml(file_path):
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
