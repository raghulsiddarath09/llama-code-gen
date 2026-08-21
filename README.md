# Python code generation — LLaMA 3.2 3B + QLoRA

Fine-tuning LLaMA 3.2 3B for Python code generation on a free-tier T4, with an
evaluation harness built to catch its own errors.

**Model:** [Raghul09/llama-code-gen-lora](https://huggingface.co/Raghul09/llama-code-gen-lora) · **Demo:** [Space](https://huggingface.co/spaces/Raghul09/llama-code-gen)

## Result

| | Base | Fine-tuned |
|---|---|---|
| HumanEval pass@1 (clean subset, n=37) | 40.5% | **54.1%** |
| Valid Python, free-form instruction (n=40) | 57.5% | **100%** |
| pass@1, signature specified (n=40) | 82.5% | **90%** |

A 9.2M-parameter adapter — 0.285% of the model — trained in 1.88 hours.

## What the base model actually got wrong

The premise going in was that the base model couldn't follow instructions. Running
it first showed otherwise: it followed instructions, produced correct code, and
stopped cleanly. The failure was **language selection** — 15 of 40 prompts came back
as JavaScript or Java, and 2 more as Python wrapped in markdown fences.

Fine-tuning's contribution is output conformance, not capability. With a function
signature specified in the prompt, the base model already reaches 97.5% valid Python
and 82.5% pass@1.

## Memory

Measured, each configuration in a fresh process:

| Configuration | Batch | Peak VRAM | Result |
|---|---|---|---|
| Full fine-tuning, BF16 | 1 | 15.48 GB | **OOM** |
| LoRA on BF16 base | 4 | 14.33 GB | fit, 92% utilization |
| QLoRA 4-bit NF4 | 4 | **4.17 GB** | fit |

Quantization alone: **3.4x reduction**, with LoRA config and batch size held constant.

## Rank sweep

| Rank | Trainable | Val loss | Valid Python | pass@1 |
|---|---|---|---|---|
| 8 | 4,587,520 | 0.4795 | 100% | 82.5% |
| 16 | 9,175,040 | 0.4753 | 100% | 90.0% |
| 32 | 18,350,080 | 0.4739 | 100% | 82.5% |

4x the parameters bought 1.2% lower validation loss and no consistent pass@1 gain.
The target behaviour is low-rank.

## Latency

103 ms/token steady state, 9.7 tokens/sec, 5.2 s median for a ~49-token function
(T4, 4-bit, greedy). Slow by design — quantization trades throughput for memory.

## Measurement errors found and corrected

Five, each caught by inspecting individual data points rather than accepting an
aggregate:

1. **Training data contamination.** A keyword filter kept 58.7% of CodeAlpaca as
   "Python." AST-parsing a 300-example sample showed 38% weren't — mostly Java and
   JavaScript matching on shared `for` and `class`. Replaced with `ast.parse` plus a
   syntax-tree check: 0% on re-check.

2. **Packing silently disabled loss masking.** Batch inspection before training showed
   1% of positions masked instead of ~87%, and 1022-token sequences instead of 256.
   Packing needs Flash Attention for block-diagonal masking, which needs Ampere; the
   T4 is Turing.

3. **Naming confound.** Initial pass@1 read 35% base / 50% fine-tuned. 17 of 20
   failures were `NameError` — correct code under a different function name than the
   test called. Specifying signatures corrected the baseline by 47 points.

4. **Indentation destruction.** Raw-format HumanEval returned 0% for both models. The
   fence-stripper called `.strip()`, removing leading indentation from function bodies.
   All 50 failures were `IndentationError`.

5. **Memory high-water contamination.** Measuring all three configs in one process gave
   12.95 GB for QLoRA instead of 4.17. `max_memory_reserved` is a process-lifetime
   high-water mark that `reset_peak_memory_stats` does not clear.

## Benchmark contamination

13 of 50 HumanEval problems have their function names defined in CodeAlpaca. The
fine-tuned model scores 69.2% on those vs ~52% on clean problems, under both prompt
formats. Headline numbers use the clean subset. Name matching catches exact reuse but
misses paraphrases, so 26% is a lower bound.

## Layout

    src/data.py       filtering, splitting, prompt formatting
    src/train.py      QLoRA config and training loop
    src/generate.py   inference helpers
    src/evaluate.py   AST validity, pass@1 harness
    tests/            37 tests
    results/          all measurements as JSON

## Tests

    pip install -r requirements.txt
    python -m pytest tests/ --cov=src

37 passing. 91% coverage on GPU-free logic (`data`, `evaluate`, `generate`);
`train.py` requires a GPU and is not unit tested.

## Setup

    pip install -r requirements.txt
    huggingface-cli login   # LLaMA 3.2 is gated

## Limitations

CodeAlpaca-20K is GPT-generated and unverified, so model quality is bounded by the
teacher. Python only. Median training example was 89 tokens; long generations degrade.
Known failure modes: repetition loops causing truncation, and calling helper functions
it never defines.
