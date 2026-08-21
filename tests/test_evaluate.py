"""Tests for the evaluation harness."""
import pytest
from src.evaluate import strip_fences, is_valid_python, run_test, specify_signature


class TestStripFences:
    def test_removes_python_fence(self):
        assert strip_fences("```python\ndef f(): pass\n```").rstrip() == "def f(): pass"

    def test_removes_js_fence(self):
        assert strip_fences("```js\nlet x = 1;\n```").rstrip() == "let x = 1;"

    def test_handles_unclosed_fence(self):
        assert "def f" in strip_fences("```python\ndef f(): pass")

    def test_passes_through_unfenced(self):
        assert strip_fences("def f(): pass") == "def f(): pass"

    def test_preserves_body_indentation(self):
        # Regression: an earlier version called .strip() on the body, which
        # destroyed leading indentation and produced IndentationError.
        out = strip_fences("```python\n    if n < 2:\n        return False\n```")
        assert out.startswith("    if n < 2")


class TestIsValidPython:
    def test_accepts_python(self):
        assert is_valid_python("def f(x):\n    return x")

    def test_accepts_fenced_python(self):
        assert is_valid_python("```python\ndef f(x):\n    return x\n```")

    def test_rejects_javascript(self):
        assert not is_valid_python("function f(x) { return x; }")

    def test_rejects_java(self):
        assert not is_valid_python("public class C { private int x; }")


class TestRunTest:
    def test_passing_case(self):
        ok, err = run_test("def add(a,b):\n    return a+b", "assert add(2,3) == 5")
        assert ok and err is None

    def test_wrong_answer_gives_assertion_error(self):
        ok, err = run_test("def add(a,b):\n    return a-b", "assert add(2,3) == 5")
        assert not ok and "AssertionError" in err

    def test_wrong_name_gives_name_error(self):
        # The confound: correct code, different name. 17/20 of the first
        # pass@1 run's failures were this, not logic errors.
        ok, err = run_test("def plus(a,b):\n    return a+b", "assert add(2,3) == 5")
        assert not ok and "NameError" in err

    def test_syntax_error(self):
        ok, err = run_test("function f() {}", "assert True")
        assert not ok and "SyntaxError" in err

    def test_timeout_on_infinite_loop(self):
        ok, err = run_test("def f():\n    while True: pass", "f()", timeout=1)
        assert not ok


class TestSpecifySignature:
    def test_appends_function_name(self):
        out = specify_signature("Flatten a list", "def flatten(lst):\n    return lst")
        assert "flatten" in out and "Flatten a list" in out

    def test_appends_class_name(self):
        out = specify_signature("Build a stack", "class Stack:\n    pass")
        assert "Stack" in out

    def test_passthrough_when_no_definition(self):
        assert specify_signature("Do a thing", "x = 1") == "Do a thing"
