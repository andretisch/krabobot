"""Push-to-talk hotkey (hold-to-talk) for Windows / cross-platform."""

from __future__ import annotations

import threading
from dataclasses import dataclass


def parse_hotkey(spec: str) -> frozenset[str]:
    """Parse ``ctrl+alt+space`` into a frozenset of normalized key names."""
    parts = [p.strip().lower() for p in (spec or "").split("+") if p.strip()]
    aliases = {
        "control": "ctrl",
        "ctl": "ctrl",
        "option": "alt",
        "cmd": "cmd",
        "win": "cmd",
        "super": "cmd",
        "esc": "esc",
        "escape": "esc",
        "return": "enter",
        " ": "space",
    }
    out: set[str] = set()
    for p in parts:
        out.add(aliases.get(p, p))
    if not out:
        out = {"ctrl", "alt", "space"}
    return frozenset(out)


def _normalize_key(key: object) -> str | None:
    """Map pynput key object to our hotkey token."""
    try:
        from pynput.keyboard import Key, KeyCode
    except ImportError:
        return None

    if isinstance(key, KeyCode):
        if key.char:
            ch = key.char.lower()
            if ch == " ":
                return "space"
            return ch
        # Some layouts expose vk without char for Space
        if getattr(key, "vk", None) == 32:
            return "space"
        return None

    mapping = {
        Key.ctrl: "ctrl",
        Key.ctrl_l: "ctrl",
        Key.ctrl_r: "ctrl",
        Key.alt: "alt",
        Key.alt_l: "alt",
        Key.alt_r: "alt",
        Key.alt_gr: "alt",
        Key.shift: "shift",
        Key.shift_l: "shift",
        Key.shift_r: "shift",
        Key.cmd: "cmd",
        Key.cmd_l: "cmd",
        Key.cmd_r: "cmd",
        Key.space: "space",
        Key.enter: "enter",
        Key.esc: "esc",
        Key.tab: "tab",
    }
    return mapping.get(key)


@dataclass
class PttState:
    """Thread-safe PTT pressed/released flags."""

    down: bool = False
    edge_down: bool = False
    edge_up: bool = False


class PttHotkey:
    """Listen for a hold-to-talk chord; expose ``is_down`` / edge flags."""

    def __init__(self, hotkey: str = "ctrl+alt+space") -> None:
        self.combo = parse_hotkey(hotkey)
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._state = PttState()
        self._listener = None

    @property
    def hotkey_label(self) -> str:
        order = ["ctrl", "alt", "shift", "cmd"]
        mods = [k for k in order if k in self.combo]
        rest = sorted(self.combo - set(mods))
        return "+".join(mods + rest)

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError(
                "pynput is required for PTT. Install with: pip install pynput"
            ) from e

        def on_press(key: object) -> None:
            name = _normalize_key(key)
            if name is None:
                return
            with self._lock:
                self._held.add(name)
                if self.combo.issubset(self._held):
                    if not self._state.down:
                        self._state.down = True
                        self._state.edge_down = True

        def on_release(key: object) -> None:
            name = _normalize_key(key)
            if name is None:
                return
            with self._lock:
                self._held.discard(name)
                if self._state.down and not self.combo.issubset(self._held):
                    self._state.down = False
                    self._state.edge_up = True

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def is_down(self) -> bool:
        with self._lock:
            return self._state.down

    def consume_edge_down(self) -> bool:
        with self._lock:
            if self._state.edge_down:
                self._state.edge_down = False
                return True
            return False

    def consume_edge_up(self) -> bool:
        with self._lock:
            if self._state.edge_up:
                self._state.edge_up = False
                return True
            return False

    def __enter__(self) -> PttHotkey:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def combo_matches(held: set[str] | frozenset[str], combo: frozenset[str]) -> bool:
    """Pure helper for unit tests."""
    return combo.issubset(set(held))
