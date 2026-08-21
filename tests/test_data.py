"""Tests for dataset filtering and prompt formatting."""
import pytest
from src.data import is_python, format_prompt, to_prompt_completion


class TestIsPython:
    def test_accepts_function_def(self):
        assert is_python("def foo(x):\n    return x * 2")

    def test_accepts_assignment_only(self):
        # The keyword filter missed these: no def, return, or import.
        assert is_python("a = 2 + 3\nb = a * 2\nc = b / a")

    def test_accepts_comprehension(self):
        assert is_python("squares = [i**2 for i in range(10)]")

    def test_rejects_javascript(self):
        # Passed the keyword filter on 'for' and 'let'.
        js = "function maxFrom(arr) {\n  let m = arr[0];\n  for (let i=1;i<arr.length;i++){}\n}"
        assert not is_python(js)

    def test_rejects_java(self):
        # Passed the keyword filter on 'class'.
        java = "public class Car {\n    private String make;\n}"
        assert not is_python(java)

    def test_rejects_html(self):
        assert not is_python('<body style="font-size: 16px;">')

    def test_rejects_sql(self):
        assert not is_python("SELECT * FROM table_name WHERE ISBN <> 0;")

    def test_rejects_prose(self):
        assert not is_python("A bot is a software application that automates tasks.")

    def test_rejects_empty(self):
        assert not is_python("")
        assert not is_python("   ")

    def test_rejects_bare_word(self):
        # Parses as a valid Python expression but has no code construct.
        assert not is_python("hello")


class TestFormatPrompt:
    def test_no_input_branch(self):
        p = format_prompt({"instruction": "Reverse a string", "input": ""})
        assert p == "### Instruction:\nReverse a string\n\n### Response:\n"
        assert "### Input:" not in p

    def test_input_branch(self):
        p = format_prompt({"instruction": "Fix this", "input": "x = 1"})
        assert "### Input:\nx = 1" in p
        assert p.endswith("### Response:\n")

    def test_ends_with_newline(self):
        # A missing trailing newline changes the token sequence and weakens
        # the pattern the model was trained on.
        p = format_prompt({"instruction": "t", "input": ""})
        assert p.endswith(":\n")

    def test_double_newline_before_markers(self):
        p = format_prompt({"instruction": "t", "input": ""})
        assert "\n\n### Response:" in p


class TestToPromptCompletion:
    def test_produces_two_columns(self):
        r = to_prompt_completion({"instruction": "i", "input": "", "output": "def f(): pass"})
        assert set(r) == {"prompt", "completion"}

    def test_completion_is_raw_output(self):
        r = to_prompt_completion({"instruction": "i", "input": "", "output": "def f(): pass"})
        assert r["completion"] == "def f(): pass"

    def test_prompt_has_no_completion(self):
        r = to_prompt_completion({"instruction": "i", "input": "", "output": "SECRET"})
        assert "SECRET" not in r["prompt"]
