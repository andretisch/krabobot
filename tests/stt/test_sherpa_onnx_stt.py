from pathlib import Path

from krabobot.stt.sherpa_onnx_stt import SherpaOnnxTranscriber


def test_resolve_transducer_model_from_common_filenames(tmp_path: Path) -> None:
    (tmp_path / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    (tmp_path / "encoder-epoch-99.onnx").write_bytes(b"x")
    (tmp_path / "decoder-epoch-99.onnx").write_bytes(b"x")
    (tmp_path / "joiner-epoch-99.onnx").write_bytes(b"x")

    model = SherpaOnnxTranscriber._resolve_transducer_model(tmp_path)

    assert model["tokens"].endswith("tokens.txt")
    assert "encoder" in Path(model["encoder"]).name
    assert "decoder" in Path(model["decoder"]).name
    assert "joiner" in Path(model["joiner"]).name
