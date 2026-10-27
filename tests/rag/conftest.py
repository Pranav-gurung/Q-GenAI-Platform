import pytest

from backend.core.config import Settings
from backend.rag.ingest import build_index
from tests.rag.helpers import KNOWLEDGE


@pytest.fixture(scope="session")
def demo_index(tmp_path_factory):
    """Index of the real demo knowledge base (hashing embeddings), built once per session."""
    index, corpus, _ = build_index(Settings(_env_file=None), KNOWLEDGE)
    assert index is not None, [i.code for i in corpus.errors]
    path = tmp_path_factory.mktemp("rag") / "index"
    index.save(path)
    return path
