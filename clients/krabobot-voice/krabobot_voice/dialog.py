"""Pure VoiceSession state machine (no I/O)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from krabobot_voice.commands import LocalCommand, match_utterance_command
from krabobot_voice.wake import command_after_wake, matches_wake_phrase


class Mode(str, Enum):
    IDLE = "idle"
    LISTEN = "listen"
    DIALOG = "dialog"
    MEETING = "meeting"


# --- Events -----------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """ASR text for one VAD (or PTT) utterance."""

    text: str
    source: str = "vad"  # vad | ptt
    # When local ASR is unavailable, empty text still means "upload PCM".
    # When ASR ran and returned empty (music / noise), leave False to discard.
    upload_if_empty: bool = False


@dataclass(frozen=True)
class Timeout:
    """Listen / follow-up deadline fired with no speech started."""


@dataclass(frozen=True)
class TurnDone:
    """Server turn finished (TTS played or failed)."""

    ok: bool
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hotkey:
    """Hardware hotkey edge."""

    kind: str  # ptt | meeting


Event = Union[Segment, Timeout, TurnDone, Hotkey]


# --- Effects ----------------------------------------------------------------


@dataclass(frozen=True)
class Beep:
    count: int


@dataclass(frozen=True)
class SendAudio:
    """Driver should upload the latest PCM segment."""


@dataclass(frozen=True)
class SendText:
    text: str


@dataclass(frozen=True)
class StartMeeting:
    pass


@dataclass(frozen=True)
class StopMeeting:
    pass


@dataclass(frozen=True)
class RunTest:
    """Print a local test confirmation line on the client terminal."""


@dataclass(frozen=True)
class SetDeadline:
    """Seconds from now for the next listen window; None = wait forever (idle)."""

    seconds: float | None


Effect = Union[Beep, SendAudio, SendText, StartMeeting, StopMeeting, RunTest, SetDeadline]


@dataclass
class VoiceSessionConfig:
    """Tunables the state machine needs (mirrors talk.* / wake.*)."""

    listen_timeout_s: float = 10.0
    follow_up_s: float = 10.0
    wake_phrases: list[str] = field(default_factory=list)
    wake_greetings: list[str] = field(default_factory=list)
    cmd_meeting_start: list[str] = field(default_factory=list)
    cmd_meeting_stop: list[str] = field(default_factory=list)
    cmd_run_test: list[str] = field(default_factory=list)
    cmd_exit: list[str] = field(default_factory=list)
    meeting_enabled: bool = True


class VoiceSession:
    """Idle → Listen → Dialog (+ Meeting) without touching mic/HTTP."""

    def __init__(self, cfg: VoiceSessionConfig) -> None:
        self.cfg = cfg
        self.mode = Mode.IDLE
        # While MEETING: after wake-only, wait for the next command segment.
        self._meeting_listen = False

    # -- public --------------------------------------------------------------

    def client_mode(self) -> str:
        return self.mode.value

    def meeting_state(self) -> str:
        return "recording" if self.mode is Mode.MEETING else "idle"

    def on_event(self, event: Event) -> list[Effect]:
        if isinstance(event, Hotkey):
            return self._on_hotkey(event)
        if isinstance(event, Timeout):
            return self._on_timeout()
        if isinstance(event, TurnDone):
            return self._on_turn_done(event)
        if isinstance(event, Segment):
            return self._on_segment(event)
        return []

    # -- hotkey --------------------------------------------------------------

    def _on_hotkey(self, event: Hotkey) -> list[Effect]:
        kind = (event.kind or "").strip().lower()
        if kind == "meeting":
            if self.mode is Mode.MEETING:
                return self._enter_idle(beep=1, stop_meeting=True)
            if not self.cfg.meeting_enabled:
                return []
            self.mode = Mode.MEETING
            self._meeting_listen = False
            return [StartMeeting(), SetDeadline(None)]
        if kind == "ptt":
            # Driver records while held, then feeds Segment(source="ptt").
            # Arm listen so the next segment is trusted without wake.
            if self.mode is Mode.IDLE:
                self.mode = Mode.LISTEN
                return [Beep(2), SetDeadline(self.cfg.listen_timeout_s)]
            return [Beep(2)]
        return []

    # -- timeout -------------------------------------------------------------

    def _on_timeout(self) -> list[Effect]:
        if self.mode is Mode.LISTEN:
            return self._enter_idle(beep=1)
        if self.mode is Mode.DIALOG:
            return self._enter_idle(beep=1)
        if self.mode is Mode.MEETING and self._meeting_listen:
            self._meeting_listen = False
            return [SetDeadline(None)]
        # IDLE / MEETING unarmed: ignore (should not be scheduled).
        return [SetDeadline(None)]

    # -- turn done -----------------------------------------------------------

    def _on_turn_done(self, event: TurnDone) -> list[Effect]:
        effects: list[Effect] = []
        actions = tuple(event.actions or ())

        if "meeting_stop" in actions and self.mode is Mode.MEETING:
            return self._enter_idle(beep=1, stop_meeting=True)
        if "end_dialog" in actions:
            return self._enter_idle(beep=1)
        if "meeting_start" in actions:
            if not self.cfg.meeting_enabled:
                return self._enter_idle(beep=1)
            self.mode = Mode.MEETING
            self._meeting_listen = False
            effects.append(StartMeeting())
            effects.append(SetDeadline(None))
            return effects

        if "run_test" in actions:
            effects.append(RunTest())

        if not event.ok:
            return effects + self._enter_idle(beep=1)

        # Successful reply → dialog follow-up (unless disabled).
        if float(self.cfg.follow_up_s) > 0:
            if self.mode is Mode.MEETING:
                self._meeting_listen = False
                return effects + [SetDeadline(None)]
            self.mode = Mode.DIALOG
            self._meeting_listen = False
            return effects + [SetDeadline(float(self.cfg.follow_up_s))]

        return effects + self._enter_idle(beep=1)

    # -- segment -------------------------------------------------------------

    def _on_segment(self, event: Segment) -> list[Effect]:
        text = (event.text or "").strip()
        source = (event.source or "vad").strip().lower()
        upload_if_empty = bool(event.upload_if_empty)

        if self.mode is Mode.IDLE:
            return self._segment_idle(
                text, source=source, upload_if_empty=upload_if_empty
            )
        if self.mode is Mode.LISTEN:
            return self._segment_execute(
                text, beep_first=False, upload_if_empty=upload_if_empty
            )
        if self.mode is Mode.DIALOG:
            return self._segment_execute(
                text, beep_first=False, upload_if_empty=upload_if_empty
            )
        if self.mode is Mode.MEETING:
            return self._segment_meeting(
                text, source=source, upload_if_empty=upload_if_empty
            )
        return []

    def _segment_idle(
        self, text: str, *, source: str, upload_if_empty: bool = False
    ) -> list[Effect]:
        if source == "ptt":
            # PTT arm may have already moved to LISTEN; still accept from idle.
            return self._segment_execute(
                text, beep_first=True, beep_n=2, upload_if_empty=upload_if_empty
            )

        if not text or not self._is_wake(text):
            # Miss — stay idle, nothing to the server (music / noise).
            return []

        trailing = self._wake_trailing(text)
        effects: list[Effect] = [Beep(2)]
        if trailing.strip():
            return effects + self._execute_command(
                trailing, prefer_text=True, upload_if_empty=upload_if_empty
            )

        self.mode = Mode.LISTEN
        effects.append(SetDeadline(float(self.cfg.listen_timeout_s)))
        return effects

    def _segment_meeting(
        self, text: str, *, source: str, upload_if_empty: bool = False
    ) -> list[Effect]:
        # Local stop always wins (no wake required).
        cmd = self._match_cmd(text)
        if cmd is LocalCommand.MEETING_STOP:
            return self._enter_idle(beep=1, stop_meeting=True)

        if self._meeting_listen or source == "ptt":
            self._meeting_listen = False
            return self._segment_execute(
                text,
                beep_first=False,
                in_meeting=True,
                upload_if_empty=upload_if_empty,
            )

        if not text or not self._is_wake(text):
            return []

        trailing = self._wake_trailing(text)
        effects: list[Effect] = [Beep(2)]
        if trailing.strip():
            return effects + self._execute_command(
                trailing,
                prefer_text=True,
                in_meeting=True,
                upload_if_empty=upload_if_empty,
            )

        self._meeting_listen = True
        effects.append(SetDeadline(float(self.cfg.listen_timeout_s)))
        return effects

    def _segment_execute(
        self,
        text: str,
        *,
        beep_first: bool,
        beep_n: int = 2,
        in_meeting: bool = False,
        upload_if_empty: bool = False,
    ) -> list[Effect]:
        effects: list[Effect] = []
        if beep_first:
            effects.append(Beep(beep_n))
        effects.extend(
            self._execute_command(
                text,
                prefer_text=False,
                in_meeting=in_meeting,
                upload_if_empty=upload_if_empty,
            )
        )
        return effects

    def _execute_command(
        self,
        text: str,
        *,
        prefer_text: bool,
        in_meeting: bool = False,
        upload_if_empty: bool = False,
    ) -> list[Effect]:
        cmd = self._match_cmd(text)
        if cmd is LocalCommand.EXIT:
            return self._enter_idle(beep=1, stop_meeting=in_meeting)
        if cmd is LocalCommand.MEETING_START:
            if in_meeting:
                # Already recording — ignore start.
                return [SetDeadline(None)] if self.mode is Mode.MEETING else []
            if not self.cfg.meeting_enabled:
                return self._enter_idle(beep=1)
            self.mode = Mode.MEETING
            self._meeting_listen = False
            return [StartMeeting(), SetDeadline(None)]
        if cmd is LocalCommand.MEETING_STOP:
            if in_meeting or self.mode is Mode.MEETING:
                return self._enter_idle(beep=1, stop_meeting=True)
            # Not recording — ignore, back to idle.
            return self._enter_idle(beep=1)
        if cmd is LocalCommand.RUN_TEST:
            return self._run_test_follow_up(in_meeting=in_meeting)

        cleaned = (text or "").strip()
        if not cleaned:
            if prefer_text:
                # Wake trailing was empty — should not reach here.
                if self.mode is Mode.LISTEN:
                    return self._enter_idle(beep=1)
                return []
            # Empty local ASR: music/noise → discard. No ASR engine → upload.
            if upload_if_empty:
                return [SendAudio()]
            return []

        if prefer_text:
            return [SendText(cleaned)]
        return [SendAudio()]

    # -- helpers -------------------------------------------------------------

    def _run_test_follow_up(self, *, in_meeting: bool = False) -> list[Effect]:
        """Local run_test: print confirmation, stay dialog-capable (no beep storm)."""
        if in_meeting or self.mode is Mode.MEETING:
            self._meeting_listen = False
            return [RunTest(), SetDeadline(None)]
        if float(self.cfg.follow_up_s) > 0:
            self.mode = Mode.DIALOG
            self._meeting_listen = False
            return [RunTest(), SetDeadline(float(self.cfg.follow_up_s))]
        return [RunTest()] + self._enter_idle(beep=1)

    def _enter_idle(self, *, beep: int, stop_meeting: bool = False) -> list[Effect]:
        was_meeting = self.mode is Mode.MEETING
        self.mode = Mode.IDLE
        self._meeting_listen = False
        effects: list[Effect] = []
        if stop_meeting and was_meeting:
            effects.append(StopMeeting())
        if beep > 0:
            effects.append(Beep(beep))
        effects.append(SetDeadline(None))
        return effects

    def _is_wake(self, text: str) -> bool:
        return matches_wake_phrase(
            text,
            phrases=self.cfg.wake_phrases or None,
            greetings=self.cfg.wake_greetings or None,
        )

    def _wake_trailing(self, text: str) -> str:
        return command_after_wake(
            text,
            phrases=self.cfg.wake_phrases or None,
            greetings=self.cfg.wake_greetings or None,
        )

    def _match_cmd(self, text: str) -> LocalCommand | None:
        return match_utterance_command(
            text,
            wake_phrases=self.cfg.wake_phrases or None,
            wake_greetings=self.cfg.wake_greetings or None,
            meeting_start=self.cfg.cmd_meeting_start,
            meeting_stop=self.cfg.cmd_meeting_stop,
            run_test=self.cfg.cmd_run_test,
            exit_dialog=self.cfg.cmd_exit,
        )
