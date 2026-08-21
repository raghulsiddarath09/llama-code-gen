"""Evaluation: language validity and functional correctness."""
import ast
import signal


def strip_fences(code: str) -> str:
    """Remove markdown code fences without touching indentation.

    Do not call .strip() on the body: it destroys the leading indentation of
    function bodies and produces IndentationError on reassembly.
    """
    code = code.strip()
    if not code.startswith("```"):
        return code
    lines = code.split("\n")
    code = "\n".join(lines[1:])
    if code.rstrip().endswith("```"):
        code = code.rstrip()[:-3]
    return code


def is_valid_python(code: str) -> bool:
    """True if the generation parses as Python once fences are removed."""
    try:
        ast.parse(strip_fences(code))
        return True
    except SyntaxError:
        return False


def run_test(code: str, test: str, timeout: int = 5):
    """Execute generated code, then its assert. Returns (passed, error).

    Executes untrusted model output. Run in a disposable environment.
    The alarm guards against non-terminating generations.
    """
    code = strip_fences(code)

    def _handler(signum, frame):
        raise TimeoutError("execution exceeded timeout")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        ns = {}
        exec(code, ns)
        exec(test, ns)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        signal.alarm(0)


def specify_signature(instruction: str, reference: str) -> str:
    """Append the expected function name to an instruction.

    Without this, correct solutions fail with NameError because the model
    chose a different name than the test calls. Measured effect on the base
    model: 35% -> 82.5% pass@1.
    """
    import re
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", reference)
    if m:
        return (f"{instruction}. Name the function `{m.group(1)}` "
                f"with signature `{m.group(1)}({m.group(2)})`.")
    m = re.search(r"class\s+(\w+)", reference)
    if m:
        return f"{instruction}. Name the class `{m.group(1)}`."
    return instruction
