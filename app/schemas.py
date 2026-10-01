"""Request and routing models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Route = Literal["math", "general"]

# Caps prompt size (and so cost and latency) per request. Shows up in the
# OpenAPI schema, and FastAPI rejects longer bodies with a 422.
MAX_QUERY_CHARS = 4000


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS, description="The user's question.")

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


# What the router LLM returns, through structured output.
# Kept to two fields on purpose: every extra output token is latency the user
# waits through before the first answer token. Both fields are required
# (expression is nullable) so the schema is valid for OpenAI's strict
# JSON-schema mode. The docstring below is sent to the model as the schema
# description, so it is written for the model, not for developers.
class RouteDecision(BaseModel):
    """Routing decision for the user's question."""

    route: Route = Field(
        description=(
            "'math' if the user asks about mathematics: a calculation, a word "
            "problem with numbers, or a math concept. 'general' for everything else."
        )
    )
    expression: str | None = Field(
        description=(
            "Only when route is 'math' and the answer is a number that can be "
            "computed: ONE arithmetic expression that computes it, using numbers, "
            "+ - * / // % ** ( ), the functions sqrt, cbrt, exp, ln, log(x, base), "
            "log10, log2, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, "
            "degrees, radians, abs, round, floor, ceil, factorial, gcd, comb, perm, "
            "min, max, hypot, and the constants pi, e, tau. Write percentages as "
            "decimals (15% of 80 -> 0.15 * 80). "
            "Trig functions use radians. Null for math questions that are conceptual "
            "or symbolic, and null for general questions."
        )
    )
