import os

from app.interfaces.generator import Generator


class HFGenerator(Generator):
    """
    Backward-compatible Hugging Face generator.

    Supports both:
      HFGenerator(model_name, adapter_path, max_new_tokens, temperature)

    and keyword-style construction such as:
      HFGenerator(
          model_name=...,
          adapter_path=...,
          max_new_tokens=...,
          temperature=...
      )
    """

    def __init__(self, *args, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Defaults from environment, then overridden by kwargs/positional args.
        model_name = kwargs.pop(
            "model_name",
            os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        )

        adapter_path = kwargs.pop(
            "adapter_path",
            kwargs.pop(
                "adapter",
                os.getenv("HF_ADAPTER", ""),
            ),
        )

        max_new_tokens = kwargs.pop(
            "max_new_tokens",
            int(os.getenv("HF_MAX_NEW_TOKENS", "160")),
        )

        temperature = kwargs.pop(
            "temperature",
            float(os.getenv("HF_TEMPERATURE", "0.0")),
        )

        # Positional compatibility with the project's existing factory.py.
        if len(args) > 0 and args[0] is not None:
            model_name = args[0]
        if len(args) > 1 and args[1] is not None:
            adapter_path = args[1]
        if len(args) > 2 and args[2] is not None:
            max_new_tokens = args[2]
        if len(args) > 3 and args[3] is not None:
            temperature = args[3]

        self.torch = torch
        self.model_name = str(model_name)
        self.adapter_path = str(adapter_path) if adapter_path else ""
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        print(f"[HFGenerator] model={self.model_name}")
        print(f"[HFGenerator] adapter={self.adapter_path or 'NONE'}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
            )

        self.model.eval()

    def generate(self, question: str, evidence) -> str:
        # Accept list[str], tuple[str], or plain string for compatibility.
        if isinstance(evidence, (list, tuple)):
            context = "\n".join(f"- {item}" for item in evidence)
        else:
            context = str(evidence)

        prompt = f"""You are a grounded enterprise assistant.

Use ONLY the supplied evidence.
Preserve conditions, exceptions, dates, numbers, and negations exactly.
If the evidence is insufficient, explicitly say INSUFFICIENT EVIDENCE.
Do not speculate.

Evidence:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        if self.torch.cuda.is_available():
            inputs = {
                key: value.to(self.model.device)
                for key, value in inputs.items()
            }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()
