"""Write-path safety checks: keep credentials out of long-lived memory."""

from __future__ import annotations

import re

_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "llm_api_key": re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    # key = "literal" assignments; code such as `api_key = os.environ[...]` does not match.
    "credential_assignment": re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{16,}['\"]?(?![\w.\[(])"
    ),
}


def find_secrets(text: str) -> list[str]:
    """Return the names of secret patterns found in ``text`` (empty when clean)."""
    return [name for name, pattern in _SECRET_PATTERNS.items() if pattern.search(text)]
