# src/config.py
import os
import re
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(__file__).parent.parent / config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Expand environment variables of the form ${VAR_NAME}
        content = re.sub(
            r"\$\{([^}]+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),  # fallback to placeholder if not found
            content
        )

        return yaml.safe_load(content)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return f"Config({self.config_path})"


# Singleton pattern for configuration across the application
config = Config()
