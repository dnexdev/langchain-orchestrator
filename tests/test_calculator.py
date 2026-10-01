import time

import pytest

from app.calculator import CalculatorError, calculator, evaluate, format_number, is_plain_expression


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", "4"),
        ("(1 + 2) * 3", "9"),
        ("10 / 4", "2.5"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("2^10", "1024"),  # caret is accepted as power
        ("-3**2", "-9"),  # Python precedence: -(3**2)
        ("0.1 + 0.2", "0.3"),  # no float noise in the output
        ("2**64", "18446744073709551616"),  # exact big integers
        ("sqrt(16)", "4"),
        ("log(100, 10)", "2"),
        ("ln(e)", "1"),
        ("factorial(5)", "120"),
        ("comb(5, 2)", "10"),
        ("round(2.567, 2)", "2.57"),
        ("0.15 * 80", "12"),
        ("12 × 3 ÷ 4", "9"),  # unicode operators
        ("max(3, 9, 4)", "9"),
        ("pi", "3.14159265359"),
    ],
)
def test_evaluates(expression: str, expected: str) -> None:
    assert format_number(evaluate(expression)) == expected


@pytest.mark.parametrize(
    "expression",
    [
        # code execution attempts
        "__import__('os').system('ls')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd')",
        "eval('1')",
        "lambda: 1",
        "[1, 2]",
        "{'a': 1}",
        "'a' * 3",
        "x + 1",
        "sqrt(x=4)",
        "sqrt(*[4])",
        "1 < 2",
        "2 if 1 else 3",
        "True + 1",
        # resource exhaustion
        "9**9**9",
        "10**100000",
        "factorial(100000)",
        "comb(1000000, 500000)",
        "round(5, -1000000000)",
        "2.0**100000",
        "10**400 / 3",  # int too large to turn into a float
        "2**1024 + 0.5",
        "10**3000 * 1.0",
        "1" + "+1" * 300,
        # math errors
        "1/0",
        "sqrt(-1)",
        "(-8)**(1/3)",
        "1e308 * 10",
        "",
    ],
)
def test_rejects(expression: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_hostile_inputs_are_cheap() -> None:
    # Every rejection above should come back in well under a millisecond,
    # so a burst of hostile expressions cannot stall the event loop.
    start = time.perf_counter()
    for expression in ["9**9**9", "factorial(100000)", "comb(1000000, 500000)", "round(5, -10**9)"] * 50:
        with pytest.raises(CalculatorError):
            evaluate(expression)
    assert time.perf_counter() - start < 0.5


def test_huge_int_is_printed_in_scientific_notation() -> None:
    assert format_number(evaluate("factorial(1000)")).startswith("4.02387260077093")
    assert format_number(evaluate("factorial(1000)")).endswith("e+2567")


def test_is_plain_expression() -> None:
    assert is_plain_expression("12*(3+4)")
    assert is_plain_expression("sqrt(2)")
    assert not is_plain_expression("42")  # a bare number is not a calculation
    assert not is_plain_expression("pi")
    assert not is_plain_expression("capital of France")
    assert not is_plain_expression("1/0")  # parses, but cannot be computed


async def test_langchain_tool_wrapper() -> None:
    assert calculator.name == "calculator"
    assert "expression" in calculator.args
    assert calculator.invoke({"expression": "6 * 7"}) == "42"
    assert await calculator.ainvoke({"expression": "2^8"}) == "256"
    with pytest.raises(CalculatorError):
        await calculator.ainvoke({"expression": "__import__('os')"})
