from pathlib import Path
import yaml


class Config:
    def __init__(self, path: str = "config.yaml"):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def __getitem__(self, item):
        return self.data.get(item)

    def get(self, key, default=None):
        return self.data.get(key, default)
