from pathlib import Path

import yaml


class Config:
    def __init__(self, filename: str = "config.yaml"):
        self.base_path = Path(__file__).resolve().parent
        self.path = self.base_path / filename
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.path.absolute()}")

        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __getitem__(self, item):
        return self.data.get(item)

    def get(self, key, default=None):
        return self.data.get(key, default)
