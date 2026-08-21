"""Dataset filtering, splitting, and prompt formatting."""
import ast
from datasets import load_dataset


CODE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
              ast.AugAssign, ast.For, ast.While, ast.If, ast.Import,
              ast.ImportFrom, ast.Call, ast.ListComp, ast.DictComp, ast.Return)


def is_python(output: str, min_len: int = 10) -> bool:
    """True if `output` parses as Python and contains at least one code construct.

    Keyword matching is not sufficient: `for` and `class` appear in Java and
    JavaScript. AST parsing plus a construct check rejects both those and bare
    prose that happens to be a valid Python expression.
    """
    out = output.strip()
    if not out or len(out) < min_len:
        return False
    try:
        tree = ast.parse(out)
    except SyntaxError:
        return False
    return any(isinstance(n, CODE_NODES) for n in ast.walk(tree))


def build_text(example: dict) -> str:
    """Full training string: prompt + completion, Alpaca format."""
    return format_prompt(example) + example["output"]


def format_prompt(example: dict) -> str:
    """Prompt half only. Must match inference format exactly."""
    if example.get("input", "").strip():
        return (f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n### Response:\n")
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n"


def to_prompt_completion(example: dict) -> dict:
    """Split into prompt/completion columns so TRL can mask instruction tokens.

    A single `text` column causes completion_only_loss to mask padding only.
    """
    return {"prompt": format_prompt(example), "completion": example["output"]}


def prepare_dataset(tokenizer, max_length: int = 256, seed: int = 42):
    """CodeAlpaca -> Python-only, length-capped, 80/10/10 split."""
    raw = load_dataset("sahil2801/CodeAlpaca-20k")["train"]
    python_only = raw.filter(lambda ex: is_python(ex["output"]))
    capped = python_only.filter(
        lambda ex: len(tokenizer.encode(build_text(ex))) <= max_length)

    splits = capped.train_test_split(test_size=0.1, seed=seed)
    inner = splits["train"].train_test_split(test_size=0.111, seed=seed)
    return inner["train"], inner["test"], splits["test"]
