"""Shared text utilities."""

from __future__ import annotations

import re


def sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    return re.split(r"(?<=[.!?])\s+", text)
