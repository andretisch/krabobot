"""Workspace file download path checks."""

from pathlib import Path

from krabobot.api.web_files import content_disposition, resolve_workspace_download


def test_resolve_allows_file_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "reports" / "итог.docx"
    target.parent.mkdir()
    target.write_bytes(b"docx")
    assert resolve_workspace_download(tmp_path, str(target)) == target
    assert resolve_workspace_download(tmp_path, "reports/итог.docx") == target


def test_resolve_rejects_escape_and_sensitive_dirs(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("no", encoding="utf-8")
    assert resolve_workspace_download(tmp_path, str(outside)) is None
    assert resolve_workspace_download(tmp_path, "../secret.txt") is None

    for folder in ("identity", "sessions", "users", ".git"):
        path = tmp_path / folder / "note.txt"
        path.parent.mkdir()
        path.write_text("x", encoding="utf-8")
        assert resolve_workspace_download(tmp_path, str(path)) is None

    missing = tmp_path / "missing.docx"
    assert resolve_workspace_download(tmp_path, str(missing)) is None


def test_content_disposition_keeps_unicode_name() -> None:
    header = content_disposition('отчёт".docx')
    assert header.startswith("attachment;")
    assert "UTF-8''" in header
    assert "\r" not in header and "\n" not in header
    assert '"' not in header.split("filename*=", 1)[-1]
