# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for krabobot-voice (Windows).

Build via: clients/krabobot-voice/scripts/build_portable.ps1
Artifacts: clients/krabobot-voice/build/krabobot-voice/
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

# SPECPATH = directory containing this .spec (injected by PyInstaller).
SPECDIR = Path(SPECPATH).resolve()  # noqa: F821
CLIENT_ROOT = SPECDIR.parent
ENTRY = CLIENT_ROOT / "krabobot_voice" / "__main__.py"
EXAMPLE_CFG = CLIENT_ROOT / "config.example.yaml"
PACKAGING_README = SPECDIR / "README.md"

datas = []
binaries = []
hiddenimports = [
    "krabobot_voice",
    "krabobot_voice.app",
    "krabobot_voice.audio_io",
    "krabobot_voice.commands",
    "krabobot_voice.config",
    "krabobot_voice.dialog",
    "krabobot_voice.http_client",
    "krabobot_voice.kws",
    "krabobot_voice.link",
    "krabobot_voice.local_asr",
    "krabobot_voice.meeting",
    "krabobot_voice.meeting_worker",
    "krabobot_voice.preprocess",
    "krabobot_voice.protocol",
    "krabobot_voice.ptt",
    "krabobot_voice.segmenter",
    "krabobot_voice.silero_vad",
    "krabobot_voice.vad",
    "krabobot_voice.wake",
    "yaml",
    "numpy",
    "httpx",
    "sounddevice",
    "pynput",
    "pynput.keyboard",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "sherpa_onnx",
    "onnxruntime",
]

if EXAMPLE_CFG.is_file():
    datas.append((str(EXAMPLE_CFG), "."))
if PACKAGING_README.is_file():
    datas.append((str(PACKAGING_README), "."))

for pkg in ("sherpa_onnx",):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:  # noqa: BLE001 — build-time best effort
        print(f"WARNING: collect_all({pkg!r}) failed: {exc}", file=sys.stderr)

# onnxruntime: avoid collect_all (submodule import crashes PyInstaller isolated child).
try:
    datas += collect_data_files("onnxruntime")
except Exception as exc:  # noqa: BLE001
    print(f"WARNING: collect_data_files('onnxruntime') failed: {exc}", file=sys.stderr)
try:
    binaries += collect_dynamic_libs("onnxruntime")
except Exception as exc:  # noqa: BLE001
    print(f"WARNING: collect_dynamic_libs('onnxruntime') failed: {exc}", file=sys.stderr)
hiddenimports += ["onnxruntime", "onnxruntime.capi", "onnxruntime.capi.onnxruntime_pybind11_state"]

for pkg in ("sounddevice", "_sounddevice_data"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: collect_all({pkg!r}) failed: {exc}", file=sys.stderr)
        try:
            binaries += collect_dynamic_libs(pkg)
        except Exception as exc2:  # noqa: BLE001
            print(
                f"WARNING: collect_dynamic_libs({pkg!r}) failed: {exc2}",
                file=sys.stderr,
            )

# CFFI wrapper + PortAudio DLLs (sounddevice loads via _sounddevice_data).
hiddenimports += ["_sounddevice", "_sounddevice_data", "_cffi_backend", "cffi"]
try:
    import _sounddevice_data as _sd_data  # type: ignore[import-not-found]

    _sd_root = Path(next(iter(_sd_data.__path__)))
    _pa_dir = _sd_root / "portaudio-binaries"
    if _pa_dir.is_dir():
        for _dll in _pa_dir.glob("*.dll"):
            binaries.append((str(_dll), str(Path("_sounddevice_data") / "portaudio-binaries")))
except Exception as exc:  # noqa: BLE001
    print(f"WARNING: PortAudio DLL collect failed: {exc}", file=sys.stderr)

if sys.platform == "win32":
    hiddenimports += [
        "pyaudiowpatch",
        "krabobot_voice.mic_permission",
    ]
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all("pyaudiowpatch")
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: collect_all('pyaudiowpatch') failed: {exc}", file=sys.stderr)
        try:
            binaries += collect_dynamic_libs("pyaudiowpatch")
        except Exception as exc2:  # noqa: BLE001
            print(
                f"WARNING: collect_dynamic_libs('pyaudiowpatch') failed: {exc2}",
                file=sys.stderr,
            )
    # Bundle entire winrt/ tree (namespace + .pyd extensions). Do NOT add to
    # hiddenimports / Analysis (winrt + onnxruntime together crash PyInstaller).
    try:
        import winrt as _winrt_pkg  # type: ignore[import-not-found]

        _winrt_dir = Path(list(_winrt_pkg.__path__)[0])
        # Whole package as data keeps import layout; binaries for .pyd/.dll.
        datas.append((str(_winrt_dir), "winrt"))
        for _bin in list(_winrt_dir.glob("*.pyd")) + list(_winrt_dir.glob("*.dll")):
            binaries.append((str(_bin), "winrt"))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: winrt package collect failed: {exc}", file=sys.stderr)

a = Analysis(  # noqa: F821
    [str(ENTRY)],
    pathex=[str(CLIENT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "onnxruntime.quantization",
        "onnxruntime.transformers",
        "onnxruntime.tools",
        "onnxruntime.datasets",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="krabobot-voice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="krabobot-voice",
)
