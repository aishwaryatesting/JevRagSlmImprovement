import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CURRENT_FILE = ROOT / "artifacts" / "current_model.json"
HISTORY_FILE = ROOT / "artifacts" / "model_history.json"

DEFAULT_MODEL = {
    "version": "base",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "path": None,
}

def get_current_model():
    if not CURRENT_FILE.exists():
        return DEFAULT_MODEL.copy()

    with CURRENT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)

def promote_model(version, adapter_path, base_model, metrics=None):
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)

    model_info = {
        "version": version,
        "base_model": base_model,
        "path": adapter_path,
    }

    if metrics:
        model_info["metrics"] = metrics

    with CURRENT_FILE.open("w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    history = []
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(model_info)

    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Promoted {version}")
