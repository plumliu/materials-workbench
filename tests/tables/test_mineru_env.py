import pytest

from pdf_tree_workflow.mineru_runner import _mineru_token


def test_mineru_token_comes_only_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINERU_TOKEN", "terminal-value")
    env_file = tmp_path / ".env"
    env_file.write_text('MINERU_TOKEN="file#value"\n', encoding="utf-8")

    assert _mineru_token() == "file#value"

    env_file.unlink()
    with pytest.raises(RuntimeError, match="Missing .*\\.env"):
        _mineru_token()
