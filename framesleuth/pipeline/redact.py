"""Redaction pipeline for OCR text and sidecar-like sensitive fields.

Two tiers run before any text reaches a model or is persisted: always-on
*secret* detectors (API keys, bearer tokens, JWTs, password/token assignments)
and opt-in *PII* detectors (emails, payment-card numbers, US SSNs/phones, cloud
access keys, private-key blocks). Card numbers are validated with the Luhn
checksum so an arbitrary 16-digit string (an order id, a build number) is not
redacted — only a genuine card is.
"""

from __future__ import annotations

import re

from framesleuth.schemas import Redaction

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "api_key",
        re.compile(r"\b(sk|rk|pk)_[A-Za-z0-9]{8,}\b"),
    ),
    (
        "bearer_token",
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}\b", re.IGNORECASE),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "password_assignment",
        re.compile(r"\b(password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    ),
    (
        "token_assignment",
        re.compile(
            r"\b(token|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    ),
]

# PII detectors applied only when ``redact_pii`` is set. Kept separate from the
# secret tier so they can be disabled independently.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("us_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Require separators so a bare 10-digit identifier is not mistaken for a phone.
    ("us_phone", re.compile(r"\b(?:\+?1[ .\-])?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b")),
]

# A run of 13-19 digits, optionally grouped with spaces/dashes — a *candidate*
# card number that is only redacted once it passes the Luhn checksum.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    """Return whether ``digits`` satisfies the Luhn (mod-10) checksum."""
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_cards(text: str, timestamp: float) -> tuple[str, list[Redaction]]:
    """Redact only Luhn-valid card numbers, leaving ordinary long numbers intact."""
    redactions: list[Redaction] = []

    def _replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            redactions.append(Redaction(t=timestamp, region="credit_card", applied=True))
            return "[REDACTED]"
        return match.group()

    return _CARD_CANDIDATE.sub(_replace, text), redactions


def redact_text(
    text: str, timestamp: float = 0.0, *, redact_pii: bool = True
) -> tuple[str, list[Redaction]]:
    """Redact likely secrets (and optionally PII) from OCR/text streams.

    Args:
        text: The text to scrub.
        timestamp: Timestamp recorded on each :class:`Redaction`.
        redact_pii: Also apply the PII detectors (emails, cards, SSNs, phones,
            cloud keys). Defaults to on; pass ``False`` to redact secrets only.
    """
    redactions: list[Redaction] = []
    output = text

    for region, pattern in _SECRET_PATTERNS:
        if pattern.search(output):
            output = pattern.sub("[REDACTED]", output)
            redactions.append(Redaction(t=timestamp, region=region, applied=True))

    if redact_pii:
        # Cards first (checksum-validated), then the simple regex detectors.
        output, card_redactions = _redact_cards(output, timestamp)
        redactions.extend(card_redactions)
        for region, pattern in _PII_PATTERNS:
            if pattern.search(output):
                output = pattern.sub("[REDACTED]", output)
                redactions.append(Redaction(t=timestamp, region=region, applied=True))

    return output, redactions
