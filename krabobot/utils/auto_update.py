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


@dataclass(frozen=True)
class AutoUpdateResult:
    """Outcome of a startup auto-update attempt."""

    status: str  # disabled | skipped | up_to_date | updated | failed
    branch: str | None = None
    old_commit: str | None = None
    new_commit: str | None = None
    reason: str | None = None

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


def find_git_root(start: Path | None = None) -> Path | None:
    """Return the git work-tree root, or None if not inside a repository."""
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
        return AutoUpdateResult(status="disabled", reason="autoUpdate is false")

    if shutil.which("git") is None:
        logger.warning("Auto-update skipped: git not found on PATH")
        return AutoUpdateResult(status="skipped", reason="git not found on PATH")

    root = find_git_root(start)
    if root is None:
        logger.warning("Auto-update skipped: not a git repository")
        return AutoUpdateResult(status="skipped", reason="not a git repository")

    try:
        branch_proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root, timeout=10)
        head_proc = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=10)
        status_proc = _run_git(["status", "--porcelain"], cwd=root, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update skipped: git command timed out")
        return AutoUpdateResult(status="failed", reason="git command timed out")
    except OSError as exc:
        logger.warning("Auto-update skipped: {}", exc)
        return AutoUpdateResult(status="failed", reason=str(exc))

    if branch_proc.returncode != 0 or head_proc.returncode != 0:
        reason = (branch_proc.stderr or head_proc.stderr or "cannot read HEAD").strip()
        logger.warning("Auto-update skipped: {}", reason)
        return AutoUpdateResult(status="failed", reason=reason)

    branch = (branch_proc.stdout or "").strip() or None
    old_commit = (head_proc.stdout or "").strip() or None

    if status_proc.returncode != 0:
        reason = (status_proc.stderr or "git status failed").strip()
        logger.warning("Auto-update skipped ({}): {}", branch, reason)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
        )

    if (status_proc.stdout or "").strip():
        reason = "dirty working tree"
        logger.warning(
            "Auto-update skipped ({} @ {}): {} — leaving local changes untouched",
            branch,
            _short_sha(old_commit or ""),
            reason,
        )
        return AutoUpdateResult(
            status="skipped",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
        )

    try:
        upstream_proc = _run_git(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            cwd=root,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update skipped: git command timed out")
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason="git command timed out",
        )
    except OSError as exc:
        logger.warning("Auto-update skipped: {}", exc)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=str(exc),
        )

    if upstream_proc.returncode != 0:
        reason = "no upstream tracking branch"
        logger.warning("Auto-update skipped ({}): {}", branch, reason)
        return AutoUpdateResult(
            status="skipped",
            branch=branch,
            old_commit=old_commit,
            reason=reason,
        )

    upstream = (upstream_proc.stdout or "").strip()

    try:
        fetch_proc = _run_git(["fetch", "--quiet"], cwd=root)
        if fetch_proc.returncode != 0:
            reason = (fetch_proc.stderr or fetch_proc.stdout or "git fetch failed").strip()
            logger.warning(
                "Auto-update failed ({} @ {}): {}",
                branch,
                _short_sha(old_commit or ""),
                reason,
            )
            return AutoUpdateResult(
                status="failed",
                branch=branch,
                old_commit=old_commit,
                reason=reason,
            )

        pull_proc = _run_git(["pull", "--ff-only"], cwd=root)
        if pull_proc.returncode != 0:
            reason = (pull_proc.stderr or pull_proc.stdout or "git pull --ff-only failed").strip()
            logger.warning(
                "Auto-update failed ({} @ {} ← {}): {}",
                branch,
                _short_sha(old_commit or ""),
                upstream,
                reason,
            )
            return AutoUpdateResult(
                status="failed",
                branch=branch,
                old_commit=old_commit,
                reason=reason,
            )

        new_head = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Auto-update failed ({}): network/git timed out", branch)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason="git fetch/pull timed out",
        )
    except OSError as exc:
        logger.warning("Auto-update failed ({}): {}", branch, exc)
        return AutoUpdateResult(
            status="failed",
            branch=branch,
            old_commit=old_commit,
            reason=str(exc),
        )

    new_commit = (new_head.stdout or "").strip() or old_commit
    if new_commit == old_commit:
        logger.info(
            "Auto-update: already up to date ({} @ {} ← {})",
            branch,
            _short_sha(old_commit or ""),
            upstream,
        )
        return AutoUpdateResult(
            status="up_to_date",
            branch=branch,
            old_commit=old_commit,
            new_commit=new_commit,
        )

    logger.info(
        "Auto-update: pulled {} {} → {} (from {})",
        branch,
        _short_sha(old_commit or ""),
        _short_sha(new_commit or ""),
        upstream,
    )
    return AutoUpdateResult(
        status="updated",
        branch=branch,
        old_commit=old_commit,
        new_commit=new_commit,
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
