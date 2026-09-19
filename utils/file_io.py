from pathlib import Path

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
