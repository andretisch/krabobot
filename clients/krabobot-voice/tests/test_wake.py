"""Unit tests for wake phrase matching and VAD-segmented wake (no mic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.app import effective_wake_energy  # noqa: E402
from krabobot_voice.dialog import (  # noqa: E402
    Beep,
    Mode,
    Segment,
    VoiceSession,
    VoiceSessionConfig,
)
from krabobot_voice.segmenter import VadSegmenter  # noqa: E402
from krabobot_voice.wake import (  # noqa: E402
    command_after_wake,
    matches_wake_phrase,
    normalize_text,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Эй, Арнольд", True),
        ("эй арнольд", True),
        ("Ну эй арнольд как дела", True),
        ("Hey Arnold", True),
        ("hey, arnold!", True),
        ("эй арнолд", True),  # ASR glitch
        ("ей арнольд", True),
        ("эйарнольд", True),
        # Log spam from user — must wake
        ("Привет, Арнольд.", True),
        ("Привет, Арнольд", True),
        ("привет арнольд", True),
        ("Привет, Арнольд!", True),
        ("Привет", False),
        ("Арнольд.", False),  # name alone, no greeting
        ("эй александр", False),
        ("эй бот", False),
        ("", False),
    ],
)
def test_matches_wake_phrase(text: str, expected: bool) -> None:
    assert matches_wake_phrase(text) is expected


def test_matches_wake_with_leading_filler() -> None:
    """Full VAD utterance «Давай скажем эй арнольд» must still wake."""
    assert matches_wake_phrase(
        "Давай скажем эй арнольд",
        phrases=["Эй, Арнольд"],
    )
    assert matches_wake_phrase(
        "давай скажем эй арнольд",
        phrases=["эй арнольд", "привет арнольд"],
    )
    assert not matches_wake_phrase("Давай скажем.", phrases=["Эй, Арнольд"])


def test_custom_phrases() -> None:
    assert matches_wake_phrase("ок арнольд", phrases=["ок арнольд"])
    assert not matches_wake_phrase("привет арнольд", phrases=["ок арнольд"], greetings=[])


def test_custom_wake_phrase_ok_bot() -> None:
    phrases = ["ок бот"]
    assert matches_wake_phrase("Ок, Бот!", phrases=phrases, greetings=["ок"])
    assert matches_wake_phrase("ок ботт", phrases=phrases, greetings=["ок"])  # ASR glitch
    assert matches_wake_phrase("ну ок бот скажи", phrases=phrases, greetings=["ок"])
    assert not matches_wake_phrase("привет арнольд", phrases=phrases, greetings=["ок"])
    assert not matches_wake_phrase("бот", phrases=phrases, greetings=["ок"])  # name alone


def test_custom_wake_phrase_fuzzy_token_order() -> None:
    phrases = ["привет краб"]
    # Order-independent token cover
    assert matches_wake_phrase("краб привет", phrases=phrases, greetings=["привет"])
    assert matches_wake_phrase("приветт краб", phrases=phrases, greetings=["привет"])


def test_wake_phrase_from_config_primary() -> None:
    """wake.phrase string with comma is one phrase, not a list split."""
    from krabobot_voice.config import _resolve_wake_phrases

    out = _resolve_wake_phrases(
        {"phrase": "Эй, Арнольд"},
        {},
        ["fallback"],
    )
    assert out == ["Эй, Арнольд"]


def test_wake_phrase_merges_aliases() -> None:
    from krabobot_voice.config import _resolve_wake_phrases

    out = _resolve_wake_phrases(
        {"phrase": "Ок, Бот", "phrases": ["hey bot", "ок бот"]},
        {},
        ["fallback"],
    )
    assert out[0] == "Ок, Бот"
    assert "hey bot" in out


def test_default_talk_silence_end_and_listen_timeout() -> None:
    from krabobot_voice.config import VoiceClientConfig

    cfg = VoiceClientConfig()
    assert cfg.silence_end_s == 2.0
    assert cfg.talk_listen_timeout_s == 10.0
    assert cfg.talk_follow_up_s == 10.0
    assert cfg.wake_silence_end_s == 0.55


def test_load_config_wake_vad_params(tmp_path: Path) -> None:
    import yaml
    from krabobot_voice.config import VoiceClientConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "base_url": "http://127.0.0.1:8900",
                "wake": {
                    "phrase": "Привет, Краб",
                    "max_s": 8.0,
                    "silence_end_s": 0.5,
                    "min_speech_s": 0.4,
                    "energy_threshold": 0.006,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(cfg_path)
    assert cfg.wake_phrases[0] == "Привет, Краб"
    assert "привет" in [g.lower() for g in cfg.wake_greetings]
    assert cfg.wake_max_s == 8.0
    assert cfg.wake_silence_end_s == 0.5
    assert cfg.wake_min_speech_s == 0.4
    assert cfg.wake_energy_threshold == 0.006
    assert cfg.silence_end_s == 2.0  # Talk default when YAML omits it
    assert matches_wake_phrase(
        "привет краб",
        phrases=cfg.wake_phrases,
        greetings=cfg.wake_greetings,
    )


def test_load_config_legacy_window_s_maps_to_max_s(tmp_path: Path) -> None:
    """Old wake.window_s is accepted as wake.max_s alias."""
    import yaml
    from krabobot_voice.config import VoiceClientConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"wake": {"window_s": 4.0}}, allow_unicode=True),
        encoding="utf-8",
    )
    cfg = VoiceClientConfig.load(cfg_path)
    assert cfg.wake_max_s == 4.0


def test_normalize_yo() -> None:
    assert "е" in normalize_text("Ёжик")
    assert normalize_text("Эй, Арнольд!") == "эй арнольд"


class _FakeMic:
    """Loud PCM then silence — one VAD-closed utterance, then idle."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        block: int = 800,
        loud_blocks: int = 40,
        silence_blocks: int = 80,
    ):
        self.sample_rate = sample_rate
        self.block = block
        self._i = 0
        self._loud_blocks = loud_blocks
        self._silence_blocks = silence_blocks

    def read_block(self) -> np.ndarray:
        self._i += 1
        if self._i <= self._loud_blocks:
            return np.ones(self.block, dtype=np.int16) * 12000
        return np.zeros(self.block, dtype=np.int16)


class _FakeAsr:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0
        self.pcm_sizes: list[int] = []

    def transcribe_pcm16(self, pcm: np.ndarray, sample_rate: int = 16000, **_: object) -> str:
        self.calls += 1
        self.pcm_sizes.append(int(np.asarray(pcm).size))
        return self.text


def _wake_vad_kwargs() -> dict:
    return dict(
        energy_threshold=0.012,
        silence_end_s=0.4,
        max_s=4.0,
        min_speech_s=0.35,
        speech_start_s=0.15,
        preroll_s=0.2,
    )


def _idle_wake_loop(
    mic: object,
    asr: object,
    *,
    phrases: list[str],
    **vad_kwargs: object,
) -> tuple[str, VoiceSession]:
    """Driver-shaped idle loop: segment → ASR → VoiceSession until wake."""
    seg = VadSegmenter(
        mic.read_block,  # type: ignore[attr-defined]
        sample_rate=int(mic.sample_rate),  # type: ignore[attr-defined]
        block=int(mic.block),  # type: ignore[attr-defined]
        **vad_kwargs,  # type: ignore[arg-type]
    )
    session = VoiceSession(
        VoiceSessionConfig(
            listen_timeout_s=10.0,
            follow_up_s=10.0,
            wake_phrases=list(phrases),
            wake_greetings=["эй", "привет", "hey"],
        )
    )
    deadline = None  # idle forever until match
    for _ in range(20):
        pcm = seg.next_segment(deadline)
        if pcm is None:
            continue
        text = asr.transcribe_pcm16(pcm, sample_rate=int(mic.sample_rate))  # type: ignore[attr-defined]
        effects = session.on_event(Segment(text))
        if session.mode is not Mode.IDLE or any(isinstance(e, Beep) for e in effects):
            return text, session
    raise AssertionError("wake not reached")


def test_wait_for_wake_asr_vad_segment_wakes_once() -> None:
    """Simulate VAD-closed utterance → mock ASR → VoiceSession wake."""
    mic = _FakeMic(loud_blocks=40)
    asr = _FakeAsr("Привет, Арнольд.")
    detail, session = _idle_wake_loop(
        mic,
        asr,
        phrases=["Эй, Арнольд", "привет арнольд"],
        **_wake_vad_kwargs(),
    )
    assert session.mode is Mode.LISTEN
    assert "Арнольд" in detail
    assert asr.calls == 1
    assert asr.pcm_sizes[0] > 0


def test_wait_for_wake_asr_full_phrase_with_filler() -> None:
    """One closed VAD segment with filler+phrase must wake (no sliding hop)."""
    mic = _FakeMic(loud_blocks=50)
    asr = _FakeAsr("Давай скажем эй арнольд")
    detail, session = _idle_wake_loop(
        mic,
        asr,
        phrases=["Эй, Арнольд"],
        **_wake_vad_kwargs(),
    )
    assert session.mode is Mode.LISTEN
    assert "арнольд" in detail.lower()
    assert asr.calls == 1


def test_wait_for_wake_asr_miss_then_match_next_utterance() -> None:
    """Non-wake VAD segment is discarded; next closed utterance can wake."""

    class _ProgressiveMic(_FakeMic):
        """Utterance 1 (loud→silence) then utterance 2 (loud→silence)."""

        def __init__(self) -> None:
            super().__init__(loud_blocks=35, silence_blocks=30)
            self._phase = 0

        def read_block(self) -> np.ndarray:
            self._i += 1
            if self._i <= 35 or (60 < self._i <= 95):
                return np.ones(self.block, dtype=np.int16) * 12000
            return np.zeros(self.block, dtype=np.int16)

    class _ProgressiveAsr:
        def __init__(self) -> None:
            self.calls = 0
            self.texts_seen: list[str] = []

        def transcribe_pcm16(
            self, pcm: np.ndarray, sample_rate: int = 16000, **_: object
        ) -> str:
            self.calls += 1
            text = "шум в комнате" if self.calls == 1 else "эй арнольд"
            self.texts_seen.append(text)
            return text

    mic = _ProgressiveMic()
    asr = _ProgressiveAsr()
    detail, session = _idle_wake_loop(
        mic,
        asr,
        phrases=["Эй, Арнольд"],
        **_wake_vad_kwargs(),
    )
    assert session.mode is Mode.LISTEN
    assert "арнольд" in detail.lower()
    assert asr.calls == 2
    assert asr.texts_seen[0] == "шум в комнате"


def test_effective_wake_energy_lower_for_loopback() -> None:
    assert effective_wake_energy(0.006, listen_source="mic") == 0.006
    lb = effective_wake_energy(0.006, listen_source="loopback")
    assert lb < 0.006
    assert lb >= 0.0015


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Эй, Арнольд", ""),
        ("эй арнольд", ""),
        ("Эй, Арнольд, какая погода?", "какая погода"),
        ("эй арнольд включи свет", "включи свет"),
        ("Hey Arnold open the door", "open the door"),
        # Wake at the end of a long clip — leading filler is NOT the command.
        (
            "Попробовать протестировать агента. Давай скажем: «Эй, Арнольд!»",
            "",
        ),
        ("Давай скажем эй арнольд", ""),
        ("Ну эй арнольд как дела", "как дела"),
        ("привет арнольд начни запись", "начни запись"),
        ("шум без wake", ""),
        ("", ""),
    ],
)
def test_command_after_wake(text: str, expected: str) -> None:
    assert command_after_wake(text) == expected


def test_command_after_wake_custom_phrase() -> None:
    phrases = ["ок бот"]
    assert command_after_wake("ок бот статус", phrases=phrases, greetings=["ок"]) == "статус"
    assert command_after_wake("скажи ок бот", phrases=phrases, greetings=["ок"]) == ""
    assert (
        command_after_wake("ок бот начни совещание", phrases=phrases, greetings=["ок"])
        == "начни совещание"
    )
