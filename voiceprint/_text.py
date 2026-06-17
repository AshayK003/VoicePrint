"""Shared text utilities."""

from __future__ import annotations

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)


def sentences(text: str) -> list[str]:
    """Split text into sentences using pysBD (rule-based, handles abbreviations)."""
    return [s.strip() for s in _segmenter.segment(text)]
