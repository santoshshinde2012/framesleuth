"""Dedicated OCR backstop for error frames.

The VLM does OCR as a side task; when it misreads a small stack trace there is no
independent check. This adds a deterministic second reader — Tesseract via the
optional ``pytesseract`` extra — used only on suspected error frames whose VLM OCR
came back sparse. It is strictly additive and fail-open: if the extra (or the
``tesseract`` binary) is absent, every call returns ``None`` and the pipeline is
unchanged. Isolating OCR from the VLM also separates an OCR failure from a model
hallucination.
"""

from __future__ import annotations

from framesleuth.logging_config import get_logger

logger = get_logger("pipeline.ocr")


def ocr_available() -> bool:
    """Whether the optional OCR backstop (pytesseract) can be imported."""
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return True


def ocr_image(image_path: str) -> str | None:
    """Return Tesseract's reading of an image, or ``None`` if OCR is unavailable.

    Never raises: a missing extra, a missing ``tesseract`` binary, or an unreadable
    image all yield ``None`` so the caller simply keeps the VLM's OCR.
    """
    try:
        import cv2
        import pytesseract
    except Exception:  # pragma: no cover - exercised only without the extra
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        text = pytesseract.image_to_string(image)
        normalized = " ".join(text.split()).strip()
        return normalized or None
    except Exception as exc:  # pragma: no cover - tesseract binary missing, etc.
        logger.debug("OCR backstop skipped for %s: %s", image_path, exc)
        return None
