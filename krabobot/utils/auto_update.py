"""Optional git auto-update before ``krabobot serve`` / ``krabobot gateway`` start."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

_GIT_TIMEOUT_S = 60
_SKIP_ENV = "KRABOBOT_SKIP_AUTO_UPDATE"
_ROOT_ENV = "KRABOBOT_ROOT"


@dataclass(frozen=True)
class AutoUpdateResult:
    """Outcome of a startup auto-update attempt."""

    status: str  # disabled | skipped | up_to_date | updated | failed
    branch: str | None = None
    old_commit: str | None = None
    new_commit: str | None = None
    reason: str | None = None
    root: str | None = None

    @property
    def changed(self) -> bool:
        return self.status == "updated"


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    timeout: float = _GIT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _short_sha(sha: str) -> str:
    return sha[:7] if sha else ""


def package_code_root() -> Path:
    """Directory that contains the installed ``krabobot`` package (repo or site-packages)."""
    import krabobot

    return Path(krabobot.__file__).resolve().parent


def iter_git_search_starts(start: Path | None = None) -> list[Path]:
    """
    Candidate directories to probe for a git work-tree.

    Preference: explicit *start*, ``KRABOBOT_ROOT``, package install location, cwd.
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(raw: Path | str | None) -> None:
        if raw is None:
            return
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            return
        if path in seen:
            return
        seen.add(path)
        out.append(path)

    _add(start)
    env_root = (os.environ.get(_ROOT_ENV) or "").strip()
    if env_root:
        _add(env_root)
    try:
        _add(package_code_root())
        # Editable checkout: …/krabobot/krabobot/__init__.py → repo root is parent.
        _add(package_code_root().parent)
    except Exception:
        pass
    _add(Path.cwd())
    return out


def find_git_root(start: Path | None = None) -> Path | None:
    """Return the git work-tree root for *start*, or None if not inside a repository."""
    if shutil.which("git") is None:
        return None
    base = (start or Path.cwd()).resolve()
    try:
        proc = _run_git(["rev-parse", "--show-toplevel"], cwd=base, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    root = (proc.stdout or "").strip()
    return Path(root) if root else None


def resolve_git_root(start: Path | None = None) -> tuple[Path | None, list[str]]:
    """
    Find a git toplevel from preferred candidates.

    Returns ``(root_or_none, tried_paths_as_str)``.
    """
    tried: list[str] = []
    for candidate in iter_git_search_starts(start):
        tried.append(str(candidate))
        root = find_git_root(candidate)
        if root is not None:
            return root, tried
    return None, tried


def try_git_auto_update(
    *,
    enabled: bool,
    start: Path | None = None,
) -> AutoUpdateResult:
    """
    If *enabled*, fetch and fast-forward pull the current branch.

    Never raises for expected failure modes (not a repo, dirty tree, network,
    pull error): returns a result with ``status=\"failed\"`` or ``\"skipped\"``.
    """
    if not enabled:
        return AutoUpdateResult(status="disabled", reason="gateway.autoUpdate is false")

    if os.environ.get(_SKIP_ENV, "").strip() in {"1", "true", "yes"}:
        reason = f"skipped ({_SKIP_ENV}=1; parent process already handled update)"
        logger.info("Auto-update {}", reason)
        return AutoUpdateResult(status="skipped", reason=reason)

    if shutil.which("git") is None:
        reason = "git not found on PATH"
        logger.warning("Auto-update skipped: {}", reason)
        return AutoUpdateResult(status="skipped", reason=reason)

    root, tried = resolve_git_root(start)
    if root is None:
        tried_fmt = ", ".join(tried) if tried else "(no candidates)"
        reason = (
            "not a git repository "
            f"(tried: {tried_fmt}). "
            f"Run from a clone, set {_ROOT_ENV} to the repo path, "
            "or use an editable install (pip install -e .)."
        )
        logger.warning("Auto-update skipped: {}", reason)
        return AutoUpdateResult(status="skipped", reason=reason)

    root_s = str(root)

    try:
        branch_proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout=10)
        head_proc = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=10)
        status_proc = _run_git(["status", "--porcelain"], cwd=root, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update skipped: git command timed out (root={})", root_s)
        return AutoUpdateResult(
            status="failed",
            reason="git command timed out",
            root=root_s,
        )
    except OSError as exc:
        logger.warning("Auto-update skipped: {} (root={})", exc, root_s)
        return AutoUpdateResult(status="failed", reason=str(exc), root=root_s)

    if branch_proc.returncode != 0 or head_proc.returncode != 0:
        reason = (branch_proc.stderr or head_proc.stderr or "cannot read HEAD").strip()
        logger.warning("Auto-update skipped: {} (root={})", reason, root_s)
        return AutoUpdateResult(status="failed", reason=reason, root=root_s)

    branch = (branch_proc.stdout or "").strip() or None
    old_commit = (head_proc.stdout or "").strip() or None

    if status_proc.returncode != 0:
        reason = (status_proc.stderr or "git status failed").strip()
        logger.warning("Auto-update skipped ({}): {} (root={})", branch, reason, root_s)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
            root=root_s,
        )

    if (status_proc.stdout or "").strip():
        reason = "dirty working tree — leaving local changes untouched"
        logger.warning(
            "Auto-update skipped ({} @ {}): {} (root={})",
            branch,
            _short_sha(old_commit or ""),
            reason,
            root_s,
        )
        return AutoUpdateResult(
            status="skipped",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
            root=root_s,
        )

    try:
        upstream_proc = _run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=root,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update skipped: git command timed out (root={})", root_s)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason="git command timed out",
            root=root_s,
        )
    except OSError as exc:
        logger.warning("Auto-update skipped: {} (root={})", exc, root_s)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=str(exc),
            root=root_s,
        )

    if upstream_proc.returncode != 0:
        reason = "no upstream tracking branch (git branch -u origin/<branch>)"
        logger.warning("Auto-update skipped ({}): {} (root={})", branch, reason, root_s)
        return AutoUpdateResult(
            status="skipped",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
            root=root_s,
        )

    upstream = (upstream_proc.stdout or "").strip()

    try:
        fetch_proc = _run_git(["fetch", "--quiet"], cwd=root)
        if fetch_proc.returncode != 0:
            reason = (fetch_proc.stderr or fetch_proc.stdout or "git fetch failed").strip()
            logger.warning(
                "Auto-update failed ({} @ {}): {} (root={})",
                branch,
                _short_sha(old_commit or ""),
                reason,
                root_s,
            )
            return AutoUpdateResult(
                status="failed",
                branch=branch,
                old_commit=old_commit,
                reason=reason,
                root=root_s,
            )

        pull_proc = _run_git(["pull", "--ff-only"], cwd=root)
        if pull_proc.returncode != 0:
            reason = (pull_proc.stderr or pull_proc.stdout or "git pull --ff-only failed").strip()
            logger.warning(
                "Auto-update failed ({} @ {} ← {}): {} (root={})",
                branch,
                _short_sha(old_commit or ""),
                upstream,
                reason,
                root_s,
            )
            return AutoUpdateResult(
                status="failed",
                branch=branch,
                old_commit=old_commit,
                reason=reason,
                root=root_s,
            )

        new_head = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update failed ({}): network/git timed out (root={})", branch, root_s)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason="git fetch/pull timed out",
            root=root_s,
        )
    except OSError as exc:
        logger.warning("Auto-update failed ({}): {} (root={})", branch, exc, root_s)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=str(exc),
            root=root_s,
        )

    new_commit = (new_head.stdout or "").strip() or old_commit
    if new_commit == old_commit:
        logger.info(
            "Auto-update: already up to date ({} @ {} ← {}, root={})",
            branch,
            _short_sha(old_commit or ""),
            upstream,
            root_s,
        )
        return AutoUpdateResult(
            status="up_to_date",
            branch=branch,
            old_commit=old_commit,
            new_commit=new_commit,
            root=root_s,
        )

    logger.info(
        "Auto-update: pulled {} {} → {} (from {}, root={})",
        branch,
        _short_sha(old_commit or ""),
        _short_sha(new_commit or ""),
        upstream,
        root_s,
    )
    return AutoUpdateResult(
        status="updated",
        branch=branch,
        old_commit=old_commit,
        new_commit=new_commit,
        root=root_s,
    )


def reexec_cli() -> None:
    """Replace this process with a fresh ``python -m krabobot …`` so new code loads."""
    args = [sys.executable, "-m", "krabobot", *sys.argv[1:]]
    logger.info("Re-exec after auto-update: {}", " ".join(args))
    # Flush handlers before replacing the process image.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os.execv(sys.executable, args)


def apply_startup_auto_update(*, enabled: bool, start: Path | None = None) -> AutoUpdateResult:
    """
    Run auto-update when enabled; re-exec the CLI if the pull changed commits.

    Safe to call early in ``serve`` / ``gateway`` before binding ports.
    """
    result = try_git_auto_update(enabled=enabled, start=start)
    if result.changed:
        reexec_cli()
    return result
