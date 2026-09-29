# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for krabobot-voice (Windows).

Build via: clients/krabobot-voice/scripts/build_portable.ps1
Artifacts: clients/krabobot-voice/build/krabobot-voice/
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

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

for pkg in ("sherpa_onnx", "onnxruntime"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as exc:  # noqa: BLE001 — build-time best effort
        print(f"WARNING: collect_all({pkg!r}) failed: {exc}", file=sys.stderr)

for pkg in ("sounddevice",):
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

if sys.platform == "win32":
    hiddenimports += ["pyaudiowpatch"]
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

a = Analysis(  # noqa: F821
    [str(ENTRY)],
    pathex=[str(CLIENT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
