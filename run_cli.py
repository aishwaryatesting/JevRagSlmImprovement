import json
import sys
from app.pipeline import run_pipeline

question = " ".join(sys.argv[1:]).strip() or "Can I return opened headphones?"
print(json.dumps(run_pipeline(question), indent=2))
