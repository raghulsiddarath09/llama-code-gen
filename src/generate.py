"""Inference helpers for the fine-tuned adapter."""
import torch


def build_prompt(instruction: str) -> str:
    """Inference prompt. Must match the training format exactly.

    A mismatch — a missing newline, a different marker — weakens the pattern
    the model learned and degrades output toward base-model behaviour.
    """
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_code(instruction, model, tokenizer, max_tokens=250, greedy=True):
    """Generate a completion. Greedy by default so evaluation is reproducible."""
    prompt = build_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=not greedy,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id)
    n = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def generate_base(instruction, peft_model, tokenizer, **kw):
    """Generate with the adapter disabled — base-model behaviour.

    Uses the same loaded weights as the adapted path, which removes any
    chance of divergence between separately loaded models.
    """
    with peft_model.disable_adapter():
        return generate_code(instruction, peft_model, tokenizer, **kw)
