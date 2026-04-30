from krabobot.bus.queue import MessageBus
from krabobot.channels.base import BaseChannel
from krabobot.config.schema import TTSConfig


class _DummyChannel(BaseChannel):
    async def start(self) -> None:  # pragma: no cover - not used in tests
        return

    async def stop(self) -> None:  # pragma: no cover - not used in tests
        return

    async def send(self, msg) -> None:  # pragma: no cover - not used in tests
        return


def test_sherpa_tts_speed_bounds(monkeypatch) -> None:
    ch = _DummyChannel(config={}, bus=MessageBus())
    monkeypatch.setenv("SHERPA_TTS_SPEED", "10")
    assert ch._sherpa_tts_speed() == 2.0
    monkeypatch.setenv("SHERPA_TTS_SPEED", "0.1")
    assert ch._sherpa_tts_speed() == 0.5


def test_sherpa_tts_speed_from_config() -> None:
    ch = _DummyChannel(config={}, bus=MessageBus())
    ch.set_tts_config(TTSConfig(sherpa_speed=1.25))
    assert ch._sherpa_tts_speed() == 1.25


def test_sherpa_model_dir_from_config() -> None:
    ch = _DummyChannel(config={}, bus=MessageBus())
    ch.set_tts_config(
        TTSConfig(
            sherpa_models_dir="~/.krabobot/models/tts",
            sherpa_model_id="csukuangfj/vits-piper-ru_RU-irina-medium",
        )
    )
    assert "vits-piper-ru_RU-irina-medium" in ch._resolve_sherpa_tts_model_dir()
