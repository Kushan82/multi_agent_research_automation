import yaml
import os

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()