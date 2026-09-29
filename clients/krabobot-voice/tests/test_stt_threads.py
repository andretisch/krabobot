"""resolve_stt_num_threads: auto = max(1, cpu_count // 2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from krabobot_voice.config import VoiceClientConfig, resolve_stt_num_threads  # noqa: E402


def test_resolve_stt_num_threads_auto_half_cpus() -> None:
    assert resolve_stt_num_threads(None, cpu_count=16) == 8
    assert resolve_stt_num_threads(0, cpu_count=16) == 8
    assert resolve_stt_num_threads("", cpu_count=12) == 6
    assert resolve_stt_num_threads(None, cpu_count=1) == 1
    assert resolve_stt_num_threads(None, cpu_count=3) == 1  # 3 // 2 == 1


def test_resolve_stt_num_threads_explicit_override() -> None:
    assert resolve_stt_num_threads(4, cpu_count=16) == 4
    assert resolve_stt_num_threads(1, cpu_count=32) == 1
    assert resolve_stt_num_threads("6", cpu_count=4) == 6


def test_resolve_stt_num_threads_invalid_falls_back_to_auto() -> None:
    assert resolve_stt_num_threads("x", cpu_count=8) == 4
    assert resolve_stt_num_threads(-3, cpu_count=8) == 4


def test_load_config_stt_threads_auto_and_override(tmp_path: Path) -> None:
    import yaml

    auto_path = tmp_path / "auto.yaml"
    auto_path.write_text("base_url: http://127.0.0.1:8900\n", encoding="utf-8")
    auto_cfg = VoiceClientConfig.load(auto_path)
    # Resolved at load time to half of this machine's CPUs.
    import os

    cpus = max(1, int(os.cpu_count() or 1))
    assert auto_cfg.stt_num_threads == max(1, cpus // 2)

    ov_path = tmp_path / "ov.yaml"
    ov_path.write_text(
        yaml.safe_dump({"base_url": "http://127.0.0.1:8900", "stt_num_threads": 3}),
        encoding="utf-8",
    )
    ov_cfg = VoiceClientConfig.load(ov_path)
    assert ov_cfg.stt_num_threads == 3
