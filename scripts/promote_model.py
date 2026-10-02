import argparse
from app.model_registry import promote_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--macro-f1", type=float, default=None)
    args = parser.parse_args()

    metrics = {}
    if args.macro_f1 is not None:
        metrics["macro_f1"] = args.macro_f1

    promote_model(
        version=args.version,
        adapter_path=args.path,
        base_model=args.base_model,
        metrics=metrics or None,
    )

if __name__ == "__main__":
    main()
