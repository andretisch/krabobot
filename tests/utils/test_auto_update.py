"""Unit tests for startup git auto-update helper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from krabobot.utils.auto_update import (
    resolve_git_root,
    try_git_auto_update,
)


def _proc(code: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = code
    m.stdout = stdout
    m.stderr = stderr
    return m


def test_auto_update_disabled() -> None:
    result = try_git_auto_update(enabled=False)
    assert result.status == "disabled"
    assert "autoUpdate" in (result.reason or "")
    assert not result.changed


def test_auto_update_skip_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KRABOBOT_SKIP_AUTO_UPDATE", "1")
    result = try_git_auto_update(enabled=True)
    assert result.status == "skipped"
    assert "KRABOBOT_SKIP_AUTO_UPDATE" in (result.reason or "")
    assert not result.changed


def test_auto_update_no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: None)
    result = try_git_auto_update(enabled=True)
    assert result.status == "skipped"
    assert "git not found" in (result.reason or "")


def test_auto_update_not_a_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.delenv("KRABOBOT_ROOT", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.iter_git_search_starts",
        lambda start=None: [start or tmp_path],
    )

    def fake_run(args, **_kwargs):
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return _proc(128, stderr="fatal: not a git repository")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True, start=tmp_path)
    assert result.status == "skipped"
    assert "not a git repository" in (result.reason or "")
    assert "KRABOBOT_ROOT" in (result.reason or "")
    assert "pip install -e" in (result.reason or "")


def test_resolve_git_root_prefers_krabobot_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    monkeypatch.setenv("KRABOBOT_ROOT", str(repo))
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")

    def fake_run(args, *, cwd, **_kwargs):
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            if Path(cwd).resolve() == repo.resolve():
                return _proc(0, stdout=str(repo) + "\n")
            return _proc(128, stderr="fatal: not a git repository")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    root, tried = resolve_git_root(start=other)
    assert root == repo.resolve()
    assert str(repo.resolve()) in tried or str(repo) in tried


def test_auto_update_dirty_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.resolve_git_root",
        lambda start=None: (tmp_path, [str(tmp_path)]),
    )

    def fake_run(args, **_kwargs):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _proc(0, stdout="main\n")
        if args == ["rev-parse", "HEAD"]:
            return _proc(0, stdout="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
        if args == ["status", "--porcelain"]:
            return _proc(0, stdout=" M krabobot/cli/commands.py\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True, start=tmp_path)
    assert result.status == "skipped"
    assert result.branch == "main"
    assert "dirty working tree" in (result.reason or "")
    assert result.root == str(tmp_path)
    assert not result.changed


def test_auto_update_ff_only_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.resolve_git_root",
        lambda start=None: (tmp_path, [str(tmp_path)]),
    )
    old = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    new = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(list(args))
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _proc(0, stdout="main\n")
        if args == ["rev-parse", "HEAD"]:
            # After pull, HEAD is new.
            if any(c[:2] == ["pull", "--ff-only"] for c in calls[:-1]):
                return _proc(0, stdout=new + "\n")
            return _proc(0, stdout=old + "\n")
        if args == ["status", "--porcelain"]:
            return _proc(0, stdout="")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return _proc(0, stdout="origin/main\n")
        if args == ["fetch", "--quiet"]:
            return _proc(0)
        if args == ["pull", "--ff-only"]:
            return _proc(0, stdout="Updating aaaaaaa..bbbbbbb\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True, start=tmp_path)
    assert result.status == "updated"
    assert result.changed
    assert result.old_commit == old
    assert result.new_commit == new
    assert result.root == str(tmp_path)
    assert ["pull", "--ff-only"] in calls


def test_auto_update_up_to_date(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.resolve_git_root",
        lambda start=None: (tmp_path, [str(tmp_path)]),
    )
    sha = "cccccccccccccccccccccccccccccccccccccccc"

    def fake_run(args, **_kwargs):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _proc(0, stdout="main\n")
        if args == ["rev-parse", "HEAD"]:
            return _proc(0, stdout=sha + "\n")
        if args == ["status", "--porcelain"]:
            return _proc(0, stdout="")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return _proc(0, stdout="origin/main\n")
        if args == ["fetch", "--quiet"]:
            return _proc(0)
        if args == ["pull", "--ff-only"]:
            return _proc(0, stdout="Already up to date.\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True, start=tmp_path)
    assert result.status == "up_to_date"
    assert not result.changed
    assert result.root == str(tmp_path)


def test_auto_update_pull_failure_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.resolve_git_root",
        lambda start=None: (tmp_path, [str(tmp_path)]),
    )

    def fake_run(args, **_kwargs):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _proc(0, stdout="main\n")
        if args == ["rev-parse", "HEAD"]:
            return _proc(0, stdout="dddddddddddddddddddddddddddddddddddddddd\n")
        if args == ["status", "--porcelain"]:
            return _proc(0, stdout="")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return _proc(0, stdout="origin/main\n")
        if args == ["fetch", "--quiet"]:
            return _proc(0)
        if args == ["pull", "--ff-only"]:
            return _proc(1, stderr="fatal: Not possible to fast-forward")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True, start=tmp_path)
    assert result.status == "failed"
    assert not result.changed
    assert "fast-forward" in (result.reason or "").lower()


def test_auto_update_uses_package_root_when_cwd_not_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editable install: cwd may be home, but package lives inside the git checkout."""
    monkeypatch.delenv("KRABOBOT_SKIP_AUTO_UPDATE", raising=False)
    monkeypatch.delenv("KRABOBOT_ROOT", raising=False)
    repo = tmp_path / "krabobot-repo"
    pkg = repo / "krabobot"
    cwd = tmp_path / "elsewhere"
    repo.mkdir()
    pkg.mkdir()
    cwd.mkdir()
    monkeypatch.setattr("krabobot.utils.auto_update.shutil.which", lambda _n: "git")
    monkeypatch.setattr(
        "krabobot.utils.auto_update.package_code_root",
        lambda: pkg,
    )
    monkeypatch.chdir(cwd)

    def fake_run(args, *, cwd, **_kwargs):
        cwd_p = Path(cwd).resolve()
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            if cwd_p == pkg.resolve() or cwd_p == repo.resolve():
                return _proc(0, stdout=str(repo) + "\n")
            return _proc(128, stderr="fatal: not a git repository")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _proc(0, stdout="main\n")
        if args == ["rev-parse", "HEAD"]:
            return _proc(0, stdout="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\n")
        if args == ["status", "--porcelain"]:
            return _proc(0, stdout="")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return _proc(0, stdout="origin/main\n")
        if args == ["fetch", "--quiet"]:
            return _proc(0)
        if args == ["pull", "--ff-only"]:
            return _proc(0, stdout="Already up to date.\n")
        raise AssertionError(f"unexpected git args: {args} cwd={cwd}")

    monkeypatch.setattr("krabobot.utils.auto_update._run_git", fake_run)
    result = try_git_auto_update(enabled=True)
    assert result.status == "up_to_date"
    assert result.root == str(repo)


def test_gateway_auto_update_default_false() -> None:
    from krabobot.config.schema import Config

    assert Config().gateway.auto_update is False
