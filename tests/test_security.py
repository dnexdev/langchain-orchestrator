"""Secrets hygiene: config comes from the environment and nothing leaks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from app.config import ConfigError, Settings

ROOT = Path(__file__).resolve().parents[1]

# Shapes of real provider keys. Short fakes in tests do not match.
KEY_PATTERNS = [
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI / Anthropic
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),  # Google
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq
    re.compile(r"lsv2_[A-Za-z0-9_]{20,}"),  # LangSmith
]


def repo_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        return [ROOT / f for f in out]
    except (subprocess.CalledProcessError, FileNotFoundError):
        skip = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.parts) and p.name != ".env"]


def test_no_api_keys_committed() -> None:
    offenders = []
    for path in repo_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in KEY_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} matches {pattern.pattern}")
    assert offenders == []


def test_env_file_is_ignored_and_example_has_no_values() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.endswith("_API_KEY=") or "_API_KEY=" not in line:
            continue
        pytest.fail(f".env.example must not contain a key value: {line.split('=')[0]}")


def test_missing_key_fails_fast_and_names_only_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        Settings(router_model="openai:gpt-5.6-luna").require_provider_keys()


def test_present_key_passes_and_value_never_appears(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "value-that-must-not-leak")
    settings = Settings(service_api_key="also-secret")
    settings.require_provider_keys()
    assert "also-secret" not in repr(settings)


def test_settings_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANSWER_MODEL", "anthropic:claude-haiku-4-5")
    monkeypatch.setenv("SPECULATIVE_ROUTING", "false")
    settings = Settings.from_env()
    assert settings.answer_model == "anthropic:claude-haiku-4-5"
    assert settings.speculative_routing is False
