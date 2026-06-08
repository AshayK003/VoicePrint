"""conftest.py — Mock heavy ML libraries before any test imports them.

torch DLL crashes on some Windows setups during collection. This file
pre-populates sys.modules with lightweight stubs so pure-Python
tests can run without loading GPU libraries.
"""

import sys
import types
from unittest.mock import MagicMock


class _StubModule(types.ModuleType):
    """A stub module that returns MagicMock for any attribute access."""

    def __getattr__(self, name):
        return MagicMock()


def _make_stub(name: str) -> _StubModule:
    mod = _StubModule(name)
    mod.__path__ = []  # makes it a package
    mod.__spec__ = None
    return mod


# Only stub out torch/transformers/sentence_transformers if torch
# is not already loaded (prevents interference when torch works fine).
if "torch" not in sys.modules:
    for mod_name in [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.utils",
        "torch.utils.data",
        "transformers",
        "transformers.generation",
        "transformers.generation.logits_process",
        "transformers.generation.configuration_utils",
        "transformers.configuration_utils",
        "sentence_transformers",
        "sentence_transformers.backend",
        "sentence_transformers.backend.load",
        "sklearn",
        "sklearn.metrics",
        "sklearn.metrics.pairwise",
    ]:
        sys.modules[mod_name] = _make_stub(mod_name)

    # Make SentenceTransformer callable and return mock
    st = sys.modules["sentence_transformers"]
    st.SentenceTransformer = MagicMock(return_value=MagicMock())

    # Make sklearn.metrics.pairwise.cosine_similarity return a value
    sk = sys.modules["sklearn.metrics.pairwise"]
    sk.cosine_similarity = MagicMock(return_value=[[0.85]])
