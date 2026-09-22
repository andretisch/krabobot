"""Shell execution tool."""

import asyncio
import os
import re
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from krabobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from krabobot.bus.queue import MessageBus


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        internal_url_allowlist: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        bus: "MessageBus | None" = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.internal_url_allowlist = internal_url_allowlist or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.bus = bus
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"
        self._user_id: str | None = None
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}

    def set_context(self, channel: str, chat_id: str, user_id: str | None = None) -> None:
        """Set the origin context for background-exec completion callbacks."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        base = f"{channel}:{chat_id}"
        self._session_key = f"user:{user_id}:{base}" if user_id else base
        self._user_id = user_id

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000
    _BG_ANNOUNCE_OUTPUT = 4_000

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return its output. "
            "Set background=true for long-running commands: returns immediately and "
            "notifies this session via a system message when the process exits."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. For foreground: default 60, max 600. "
                        "For background: optional; omit to wait until the process exits "
                        "(no 600s cap)."
                    ),
                    "minimum": 1,
                },
                "background": {
                    "type": "boolean",
                    "description": (
                        "If true, run the command in the background inside the gateway "
                        "process and announce completion into this session when it exits. "
                        "Use for long jobs (e.g. STT); prefer over cron for same-session "
                        "callbacks."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for a background job (for display)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        background: bool = False,
        label: str | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        if background:
            return await self._start_background(command, cwd, timeout, label)

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)
        return await self._run_foreground(command, cwd, effective_timeout)

    async def _start_background(
        self,
        command: str,
        cwd: str,
        timeout: int | None,
        label: str | None,
    ) -> str:
        if self.bus is None:
            return (
                "Error: background exec requires the message bus "
                "(available in the main agent / gateway, not in subagents)."
            )

        job_id = str(uuid.uuid4())[:8]
        display_label = label or (command[:40] + ("..." if len(command) > 40 else ""))
        origin = {"channel": self._origin_channel, "chat_id": self._origin_chat_id}
        session_key = self._session_key
        user_id = self._user_id

        bg_task = asyncio.create_task(
            self._run_background_job(
                job_id, display_label, command, cwd, timeout, origin, user_id=user_id
            )
        )
        self._running_tasks[job_id] = bg_task
        self._session_tasks.setdefault(session_key, set()).add(job_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(job_id, None)
            if ids := self._session_tasks.get(session_key):
                ids.discard(job_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Background exec [{}] started: {}", job_id, display_label)
        return (
            f"Background exec [{display_label}] started (id: {job_id}). "
            "I'll notify you when it completes."
        )

    async def _run_background_job(
        self,
        job_id: str,
        label: str,
        command: str,
        cwd: str,
        timeout: int | None,
        origin: dict[str, str],
        *,
        user_id: str | None = None,
    ) -> None:
        logger.info("Background exec [{}] running: {}", job_id, label)
        process: asyncio.subprocess.Process | None = None
        try:
            env = os.environ.copy()
            if self.path_append:
                env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                if timeout is not None:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )
                else:
                    stdout, stderr = await process.communicate()
            except asyncio.TimeoutError:
                await self._kill_process(process)
                await self._announce_result(
                    job_id,
                    label,
                    command,
                    f"Error: Command timed out after {timeout} seconds",
                    origin,
                    "error",
                    exit_code=None,
                    user_id=user_id,
                )
                return
            except asyncio.CancelledError:
                await self._kill_process(process)
                raise

            stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            exit_code = process.returncode
            status = "ok" if exit_code == 0 else "error"
            body = self._format_job_output(stdout_text, stderr_text, exit_code)
            logger.info("Background exec [{}] finished with exit {}", job_id, exit_code)
            await self._announce_result(
                job_id, label, command, body, origin, status, exit_code=exit_code, user_id=user_id
            )

        except asyncio.CancelledError:
            logger.info("Background exec [{}] cancelled", job_id)
            raise
        except Exception as e:
            logger.error("Background exec [{}] failed: {}", job_id, e)
            await self._announce_result(
                job_id,
                label,
                command,
                f"Error executing command: {e}",
                origin,
                "error",
                exit_code=None,
                user_id=user_id,
            )

    async def _announce_result(
        self,
        job_id: str,
        label: str,
        command: str,
        result: str,
        origin: dict[str, str],
        status: str,
        *,
        exit_code: int | None,
        user_id: str | None = None,
    ) -> None:
        if self.bus is None:
            return

        from krabobot.bus.events import InboundMessage

        status_text = "completed successfully" if status == "ok" else "failed"
        exit_line = f"Exit code: {exit_code}\n" if exit_code is not None else ""
        announce_content = f"""[Background exec '{label}' {status_text}]

Command: {command}
{exit_line}
Output:
{result}

Continue the task using this result (e.g. read saved artifacts, summarize for the user). \
Keep the user update brief. Do not mention technical details like job IDs unless asked."""

        msg = InboundMessage(
            channel="system",
            sender_id="background_exec",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            user_id=user_id,
        )
        await self.bus.publish_inbound(msg)
        logger.debug(
            "Background exec [{}] announced to {}:{}",
            job_id,
            origin["channel"],
            origin["chat_id"],
        )

    def _format_job_output(
        self, stdout_text: str, stderr_text: str, exit_code: int | None
    ) -> str:
        parts: list[str] = []
        if stdout_text:
            parts.append(stdout_text)
        if stderr_text.strip():
            parts.append(f"STDERR:\n{stderr_text}")
        if exit_code is not None:
            parts.append(f"\nExit code: {exit_code}")
        result = "\n".join(parts) if parts else "(no output)"
        return self._truncate(result, self._BG_ANNOUNCE_OUTPUT)

    async def _run_foreground(self, command: str, cwd: str, effective_timeout: int) -> str:
        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                await self._kill_process(process)
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"
            return self._truncate(result, self._MAX_OUTPUT)

        except Exception as e:
            return f"Error executing command: {str(e)}"

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        finally:
            if sys.platform != "win32":
                try:
                    os.waitpid(process.pid, os.WNOHANG)
                except (ProcessLookupError, ChildProcessError) as e:
                    logger.debug("Process already reaped or not found: {}", e)

    @staticmethod
    def _truncate(result: str, max_len: int) -> str:
        if len(result) <= max_len:
            return result
        half = max_len // 2
        return (
            result[:half]
            + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
            + result[-half:]
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all background exec jobs for the given session. Returns count cancelled."""
        tasks = [
            self._running_tasks[tid]
            for tid in self._session_tasks.get(session_key, [])
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        ]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running background exec jobs."""
        return len(self._running_tasks)

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from krabobot.security.network import contains_internal_url
        if contains_internal_url(cmd, self.internal_url_allowlist):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command) # POSIX: /absolute only
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command) # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths
