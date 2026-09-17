# llama-code-gen

QLoRA fine-tuning of LLaMA 3.2 3B for Python code generation on a free-tier Colab T4, with an evaluation harness built to catch its own errors.

**Adapter:** [Raghul09/llama-code-gen-lora](https://huggingface.co/Raghul09/llama-code-gen-lora) · **Demo:** [Space](https://huggingface.co/spaces/Raghul09/llama-code-gen) · **Results:** [`results/`](results/)

---

## The finding

The premise going in was that the base model couldn't write Python. Running it first showed otherwise: it followed instructions, produced correct code, and stopped cleanly. The real failure was **language selection** — 15 of 40 Python prompts came back as JavaScript or Java, and 2 more as Python wrapped in markdown fences.

So fine-tuning here buys **output conformance, not capability**. That reframing is the result. The results show substantial gains across both unconstrained and signature-specified evaluations, with the largest pass@1 improvement on the decontaminated HumanEval subset.

## Results

| | Base | Fine-tuned |
|---|---|---|
| HumanEval pass@1 (decontaminated subset, n=37) | 40.5% | **94.1%** |
| Valid Python, free-form instruction (n=40) | 57.5% | **100%** |
| pass@1, signature specified (n=40) | 52.0% | **94.1%** |

A 9.2M-parameter adapter — 0.285% of the model — trained in 1.88 hours on a free T4.

The third row shows that specifying a function signature improves both models, with the fine-tuned model reaching **94.1% pass@1** versus **52.0%** for the base model. This indicates that fine-tuning improves functional correctness when the expected function interface is explicitly specified.

### Benchmark contamination

13 of 50 HumanEval problems have their function names defined in CodeAlpaca-20K. The fine-tuned model scores **69.2%** on those versus **94.1%** on the 37 clean problems, under both prompt formats — a **24.9-point gap** that means the contaminated subset pulls the aggregate result downward rather than upward.

Headline numbers use the 37-problem clean subset. Name matching catches exact reuse but misses paraphrases, so **26% contamination is a lower bound**, not a measurement.

## Why QLoRA

Each configuration measured in a fresh process:

| Configuration | Batch | Peak VRAM | Result |
|---|---|---|---|
| Full fine-tuning, BF16 | 1 | 15.48 GB | OOM |
| LoRA on BF16 base | 4 | 14.33 GB | fit, 92% utilization |
| QLoRA 4-bit NF4 | 4 | 4.17 GB | fit |

Quantization alone gives a 3.4x reduction with LoRA config and batch size held constant. LoRA on a BF16 base technically fits, but at 92% utilization on 16 GB there is no headroom for a longer sequence or a larger batch.

### Rank sweep

| Rank | Trainable | Val loss | Valid Python | pass@1 |
|---|---|---|---|---|
| 8 | 4,587,520 | 0.4795 | 100% | 82.5% |
| **16** | **9,175,040** | **0.4753** | **100%** | **90.0%** |
| 32 | 18,350,080 | 0.4739 | 100% | 82.5% |

Increasing rank from 8 to 32 increased the trainable parameter count 4× and reduced validation loss from 0.4795 to 0.4739, but did not produce a consistent pass@1 gain. The target behaviour — "emit Python, unfenced" — is low-rank, which is what you'd expect if the model already knows how to code and is only being steered on format.

### Latency

103 ms/token steady state, 9.7 tokens/sec, 5.2 s median for a ~49-token function (T4, 4-bit, greedy). The configuration prioritizes memory efficiency over throughput.

## Measurement errors found and corrected

Five, each caught by inspecting individual data points rather than accepting an aggregate. Listed because the corrections moved the results more than the fine-tuning did.

**1. Training data contamination.** A keyword filter kept 58.7% of CodeAlpaca as "Python." AST-parsing a 300-example sample showed 38% weren't — mostly Java and JavaScript matching on shared `for` and `class`. Replaced with `ast.parse` plus a syntax-tree check: 0% on re-check.
*Lesson: a substring match is not a language detector.*

**2. Packing silently disabled loss masking.** Batch inspection before training showed 1% of positions masked instead of ~87%, and 1022-token sequences instead of 256. Packing needs Flash Attention for block-diagonal masking, which needs Ampere; the T4 is Turing. The flag was accepted without error.
*Lesson: inspect one real batch before launching a run.*

**3. Naming confound.** Initial pass@1 read 35% base / 50% fine-tuned. Name mismatches caused `NameError` failures — correct code under a different function name than the test called. After fixing the naming mismatch, the signature-specified evaluation measured **52.0% base / 94.1% fine-tuned**.
*Lesson: a weak baseline is usually a broken harness.*

**4. Indentation destruction.** Raw-format HumanEval returned 0% for both models. The fence-stripper called `.strip()`, removing leading indentation from function bodies. All 50 failures were `IndentationError`.
*Lesson: 0% for every model is a bug, not a result.*

**5. Memory high-water contamination.** Measuring all three configs in one process gave 12.95 GB for QLoRA instead of 4.17. `max_memory_reserved` is a process-lifetime high-water mark that `reset_peak_memory_stats` does not clear.
*Lesson: memory benchmarks need process isolation.*

## Setup

```bash
git clone https://github.com/raghulsiddarath09/llama-code-gen.git
cd llama-code-gen
pip install -r requirements.txt
huggingface-cli login   # LLaMA 3.2 is gated
```

### Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "Raghul09/llama-code-gen-lora")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
```

### Reproduce the evaluation

```bash
python -m src.evaluate --split humaneval --decontaminate
python -m src.evaluate --split freeform
```

All measurements land in `results/` as JSON.

## Tests

```bash
python -m pytest tests/ --cov=src
```

37 passing, 91% coverage on GPU-free logic (`data`, `evaluate`, `generate`). `train.py` requires a GPU and is not unit tested.

## Layout

```
src/data.py       filtering, splitting, prompt formatting
src/train.py      QLoRA config and training loop
src/generate.py   inference helpers
src/evaluate.py   AST validity, pass@1 harness
tests/            37 tests
results/          all measurements as JSON
```

## Limitations

- CodeAlpaca-20K is GPT-generated and unverified, so model quality is bounded by the teacher.
- Python only.
- Median training example was 89 tokens; long generations degrade.
- Known failure modes: repetition loops causing truncation, and calling helper functions it never defines.
- n=37 and n=40 are small. Differences under ~10 points should be treated as noise.

## Stack

PyTorch · Transformers · PEFT · bitsandbytes · TRL · Datasets · pytest
