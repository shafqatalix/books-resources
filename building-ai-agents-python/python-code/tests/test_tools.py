import pytest

pytest.importorskip("langchain")

from lib import tools  # noqa: E402


def test_list_documents_finds_sample_files():
    out = tools.list_documents.invoke({})
    assert "refund_policy.txt" in out


def test_read_document_returns_text():
    out = tools.read_document.invoke({"filename": "refund_policy.txt"})
    assert "30 days" in out


def test_read_document_blocks_path_traversal():
    out = tools.read_document.invoke({"filename": "../.env"})
    assert out.startswith("Error")


def test_read_document_unknown_file():
    out = tools.read_document.invoke({"filename": "missing.txt"})
    assert out.startswith("Error")


def test_save_note_sanitizes_title(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "NOTES_DIR", tmp_path)
    tools.save_note.invoke({"title": "../../evil name!", "text": "hello"})
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert ".." not in files[0].name
