"""QLoRA training configuration and loop."""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL = "meta-llama/Llama-3.2-3B"


def quant_config():
    """4-bit NF4 with double quantization. 6.43 GB -> 2.20 GB."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # T4 is Turing; no bf16
        bnb_4bit_use_double_quant=True)


def lora_config(r=16):
    """alpha = 2r keeps the alpha/r scaling constant across ranks, so one
    learning rate is valid for a whole rank sweep."""
    return LoraConfig(
        r=r, lora_alpha=2 * r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")


def build_model(r=16):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=quant_config(), device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return get_peft_model(model, lora_config(r))


def training_config(output_dir, epochs=3):
    """Notes on non-obvious settings:

    packing=False   — packing needs Flash Attention for block-diagonal masking,
                      which needs Ampere. On Turing it silently disables
                      completion-only loss masking.
    fp16=False      — the GradScaler calls a CUDA kernel that has no bf16
                      implementation, and adapters end up bf16 under
                      transformers v5's dtype="auto".
    max_grad_norm=0 — clipping is the only caller of scaler.unscale_().
    max_length=256  — measured: p50 89, p95 211 tokens.
    """
    return SFTConfig(
        output_dir=output_dir,
        max_length=256, packing=False, completion_only_loss=True,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4, warmup_steps=36, lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=False, bf16=False,
        gradient_checkpointing=True, max_grad_norm=0.0,
        logging_steps=25, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=3, load_best_model_at_end=True, report_to="none")


def train(model, train_ds, val_ds, output_dir, epochs=3):
    trainer = SFTTrainer(model=model, args=training_config(output_dir, epochs),
                         train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    trainer.save_model(output_dir)
    return trainer
