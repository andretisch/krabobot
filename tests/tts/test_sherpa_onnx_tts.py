from pathlib import Path

from krabobot.tts.sherpa_onnx_tts import SherpaOnnxTTS


def test_resolve_onnx_model_prefers_model_onnx(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "abc.onnx").write_bytes(b"x")
    resolved = SherpaOnnxTTS._resolve_onnx_model(tmp_path)
    assert resolved.name == "model.onnx"


def test_resolve_onnx_model_falls_back_to_any_onnx(tmp_path: Path) -> None:
    (tmp_path / "vits-piper-ru.onnx").write_bytes(b"x")
    resolved = SherpaOnnxTTS._resolve_onnx_model(tmp_path)
    assert resolved.suffix == ".onnx"
