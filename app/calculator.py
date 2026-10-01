"""A safe arithmetic evaluator exposed as a LangChain tool.

Why not `eval()` or LangChain's old LLMMathChain?
The expression we evaluate is written by an LLM, and the LLM is steered by
user text. So the expression is untrusted input. LLMMathChain passed model
output to Python evaluation and was assigned CVE-2023-29374 (arbitrary code
execution through prompt injection). Here we parse the expression into an
AST and walk it ourselves, allowing only numbers, arithmetic operators, a
fixed list of math functions and a few constants. Anything else, like names,
attributes, subscripts, lambdas or keyword arguments, is rejected before it
can run.

We also bound the cost of the few operations that can blow up
(`9**9**9`, `factorial(10**6)`), because a CPU denial of service is still a
security bug even when no code is executed.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MAX_EXPRESSION_CHARS = 256
MAX_AST_NODES = 200
# About 3,900 decimal digits. Stays under CPython's default 4,300 digit limit
# for int to str conversion, so every result we accept can also be printed.
MAX_INT_BITS = 13_000
MAX_FACTORIAL_ARG = 1_000
EXACT_INT_DIGITS = 50  # print integers exactly up to this many digits


class CalculatorError(ValueError):
    """The expression is not allowed or cannot be evaluated."""


def _check_int_size(value: Any) -> Any:
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculatorError("result is too large")
    return value


def _safe_pow(base: float, exp: float) -> float:
    # Estimate the size of an integer power before computing it, since
    # Python would happily spend minutes on 9**9**9.
    if isinstance(base, int) and isinstance(exp, int) and exp > 0 and abs(base) > 1:
        if exp * math.log2(abs(base)) > MAX_INT_BITS:
            raise CalculatorError("result is too large")
    try:
        result = operator.pow(base, exp)
    except OverflowError as exc:
        raise CalculatorError("result is too large") from exc
    if isinstance(result, complex):
        raise CalculatorError("result is not a real number")
    return result


def _safe_factorial(n: float) -> int:
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int) or n < 0:
        raise CalculatorError("factorial needs a non-negative integer")
    if n > MAX_FACTORIAL_ARG:
        raise CalculatorError(f"factorial argument must be at most {MAX_FACTORIAL_ARG}")
    return math.factorial(n)


def _log(x: float, base: float | None = None) -> float:
    return math.log(x) if base is None else math.log(x, base)


def _safe_round(x: float, ndigits: int | None = None) -> float:
    # round(5, -10**9) would build 10**(10**9) internally.
    if ndigits is not None and (not isinstance(ndigits, int) or abs(ndigits) > 100):
        raise CalculatorError("round needs an integer number of digits between -100 and 100")
    return round(x, ndigits)


def _bounded(fn: Callable[[int, int], int]) -> Callable[[int, int], int]:
    # comb(10**6, 5 * 10**5) is valid math but takes seconds to compute.
    def wrapper(n: int, k: int | None = None) -> int:
        if isinstance(n, int) and n > MAX_FACTORIAL_ARG:
            raise CalculatorError(f"{fn.__name__} argument must be at most {MAX_FACTORIAL_ARG}")
        return fn(n, k) if k is not None else fn(n)

    wrapper.__name__ = fn.__name__
    return wrapper


_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": _safe_round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "exp": math.exp,
    "ln": math.log,
    "log": _log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": _safe_factorial,
    "gcd": math.gcd,
    "comb": _bounded(math.comb),
    "perm": _bounded(math.perm),
}

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def normalize(expression: str) -> str:
    """Accept a few common ways people write math."""
    return (
        expression.strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")  # unicode minus
    )


def parse(expression: str) -> ast.Expression:
    """Parse and validate an expression without evaluating it."""
    expression = normalize(expression)
    if not expression:
        raise CalculatorError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise CalculatorError("not a valid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculatorError("expression is too complex")
    return tree


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so exclude it explicitly.
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError("only numbers are allowed")

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _eval(node.left), _eval(node.right)
        try:
            return _check_int_size(op(left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
        except OverflowError as exc:  # e.g. 10**400 / 3 (int too large for a float)
            raise CalculatorError("result is too large") from exc

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval(node.operand))

    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalculatorError(f"unknown name '{node.id}'")

    if isinstance(node, ast.Call):
        # Only a bare whitelisted name may be called: no attributes like
        # `().__class__`, no keywords, no *args.
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalculatorError("function is not allowed")
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise CalculatorError("only positional arguments are allowed")
        args = [_eval(a) for a in node.args]
        try:
            return _check_int_size(FUNCTIONS[node.func.id](*args))
        except CalculatorError:
            raise
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            raise CalculatorError(f"{node.func.id}: {exc}") from exc

    raise CalculatorError(f"{type(node).__name__} is not allowed")


def evaluate(expression: str) -> int | float:
    """Evaluate an arithmetic expression. Raises CalculatorError, nothing else."""
    try:
        result = _eval(parse(expression))
    except CalculatorError:
        raise
    except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:
        # Safety net so callers only ever have one exception type to handle.
        raise CalculatorError("could not evaluate the expression") from exc
    if isinstance(result, float) and not math.isfinite(result):
        raise CalculatorError("result is not a finite number")
    return result


def format_number(value: int | float) -> str:
    """Format a result for humans and for the LLM prompt.

    Floats are shown with 12 significant digits, so 0.1 + 0.2 prints as 0.3
    instead of 0.30000000000000004.
    """
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    if isinstance(value, int):
        digits = str(abs(value))
        if len(digits) <= EXACT_INT_DIGITS:
            return str(value)
        # Scientific notation built from the digits, since huge ints do not
        # fit in a float.
        sign = "-" if value < 0 else ""
        mantissa = (digits[0] + "." + digits[1:16]).rstrip("0").rstrip(".")
        return f"{sign}{mantissa}e+{len(digits) - 1}"
    return f"{value:.12g}"


def is_plain_expression(text: str) -> bool:
    """True if `text` is a complete expression that actually computes something.

    Used by the router fast path. A bare number ("42") or bare constant ("pi")
    does not count, since the user is probably not asking for arithmetic.
    """
    try:
        tree = parse(text)
    except CalculatorError:
        return False
    if isinstance(tree.body, (ast.Constant, ast.Name)):
        return False
    try:
        evaluate(text)
    except CalculatorError:
        return False
    return True


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "A single arithmetic expression, e.g. '(3 + 4) * 2', 'sqrt(2) / 2', '0.15 * 80', 'log(100, 10)'."
        )
    )


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the result.

    Supports + - * / // % ** and parentheses, the functions abs, round, min,
    max, sqrt, cbrt, exp, ln, log(x, base), log10, log2, trig and hyperbolic
    functions, degrees, radians, hypot, floor, ceil, factorial, gcd, comb,
    perm, and the constants pi, e, tau.
    """
    return format_number(evaluate(expression))
