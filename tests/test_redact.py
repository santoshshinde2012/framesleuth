"""Tests for secret + PII redaction."""

from __future__ import annotations

from framesleuth.pipeline.redact import _luhn_ok, redact_text


def _regions(text: str, **kwargs: bool) -> set[str]:
    _, redactions = redact_text(text, **kwargs)
    return {r.region for r in redactions}


def test_existing_secret_patterns_still_redacted() -> None:
    out, redactions = redact_text("authorization: Bearer abcdef1234567890ABCDEF")
    assert "[REDACTED]" in out
    assert any(r.region == "bearer_token" for r in redactions)


def test_private_key_block_redacted() -> None:
    out, _ = redact_text("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
    assert "PRIVATE KEY" not in out


def test_email_redacted() -> None:
    out, redactions = redact_text("contact alice.dev@example.co.uk for access")
    assert "alice.dev@example.co.uk" not in out
    assert "email" in {r.region for r in redactions}


def test_valid_credit_card_redacted_invalid_kept() -> None:
    # 4111 1111 1111 1111 is the canonical Luhn-valid Visa test number.
    redacted_valid, valid_marks = redact_text("card 4111 1111 1111 1111 on file")
    assert "4111" not in redacted_valid
    assert "credit_card" in {r.region for r in valid_marks}

    # A 16-digit order id that fails Luhn must NOT be redacted as a card.
    order = "order 1234567890123456 shipped"
    redacted_order, order_marks = redact_text(order)
    assert "1234567890123456" in redacted_order
    assert "credit_card" not in {r.region for r in order_marks}


def test_ssn_and_phone_redacted() -> None:
    assert "us_ssn" in _regions("ssn 123-45-6789")
    assert "us_phone" in _regions("call +1 415-555-0132 now")


def test_aws_access_key_redacted() -> None:
    assert "aws_access_key" in _regions("key AKIAIOSFODNN7EXAMPLE here")


def test_pii_can_be_disabled() -> None:
    out, redactions = redact_text("email me at bob@example.com", redact_pii=False)
    assert "bob@example.com" in out  # PII left intact when disabled
    assert not any(r.region == "email" for r in redactions)


def test_luhn_checksum() -> None:
    assert _luhn_ok("4111111111111111")
    assert not _luhn_ok("4111111111111112")
