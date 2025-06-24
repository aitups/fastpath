import yaml
from typing import Any

class Config:
    """
    Carga y valida la configuración centralizada desde config.yaml
    """
    def __init__(self, path: str = "config.yaml"):
        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default) 