from app.model_registry import get_current_model
import json

print(json.dumps(get_current_model(), indent=2))
