"""
command_executor.py
───────────────────
Maps detected gestures → OS-level keyboard shortcuts.

Gesture → Command mapping
─────────────────────────
  scissors    → CUT      Ctrl+X  (Cmd+X  on macOS)
  open_palm   → PASTE    Ctrl+V  (Cmd+V)
  ok_sign     → COPY     Ctrl+C  (Cmd+C)
  fist        → BOLD     Ctrl+B  (Cmd+B)
  one_finger  → ITALIC   Ctrl+I  (Cmd+I)
  call_sign   → UNDO     Ctrl+Z  (Cmd+Z)

Uses `pyautogui` for cross-platform key simulation.
On headless / CI environments it degrades gracefully with a dry-run.
"""

from __future__ import annotations

import platform
import sys
from typing import Optional

try:
    import pyautogui
    pyautogui.FAILSAFE    = True
    pyautogui.PAUSE       = 0.05
    _PYAUTOGUI_AVAILABLE  = True
except ImportError:
    _PYAUTOGUI_AVAILABLE  = False

# ─── PLATFORM DETECTION ──────────────────────────────────────────────────────────
_IS_MACOS   = platform.system() == "Darwin"
_MOD_KEY    = "command" if _IS_MACOS else "ctrl"

# ─── GESTURE → COMMAND MAP ───────────────────────────────────────────────────────
#
#   Each entry: gesture_name → {command, keys, icon, description, color}
#
COMMAND_MAP: dict[str, dict] = {
    "scissors": {
        "command":     "CUT",
        "keys":        [_MOD_KEY, "x"],
        "icon":        "✂️",
        "description": "Cut selected text / element",
        "color":       "#FF5050",
    },
    "open_palm": {
        "command":     "PASTE",
        "keys":        [_MOD_KEY, "v"],
        "icon":        "📋",
        "description": "Paste from clipboard",
        "color":       "#50DC64",
    },
    "ok_sign": {
        "command":     "COPY",
        "keys":        [_MOD_KEY, "c"],
        "icon":        "👌",
        "description": "Copy selection to clipboard",
        "color":       "#50B4FF",
    },
    "fist": {
        "command":     "BOLD",
        "keys":        [_MOD_KEY, "b"],
        "icon":        "✊",
        "description": "Toggle bold formatting",
        "color":       "#FFA050",
    },
    "one_finger": {
        "command":     "ITALIC",
        "keys":        [_MOD_KEY, "i"],
        "icon":        "☝️",
        "description": "Toggle italic formatting",
        "color":       "#DC50FF",
    },
    "call_sign": {
        "command":     "UNDO",
        "keys":        [_MOD_KEY, "z"],
        "icon":        "🤙",
        "description": "Undo last action",
        "color":       "#50F0DC",
    },
}


class CommandExecutor:
    """
    Executes OS keyboard commands mapped from gesture names.

    If pyautogui is unavailable (headless / Streamlit Cloud)
    the commands are logged but not fired — no crash.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run or not _PYAUTOGUI_AVAILABLE
        if self.dry_run:
            print("[CommandExecutor] Running in DRY-RUN mode — keys won't be pressed.")

    # ── PUBLIC API ────────────────────────────────────────────────────────────────

    def execute(self, gesture: str) -> Optional[dict]:
        """
        Fire the command linked to `gesture`.

        Returns the command dict on success, None if gesture not mapped.
        """
        cmd = COMMAND_MAP.get(gesture)
        if cmd is None:
            return None

        self._fire(cmd["keys"])
        print(f"[CommandExecutor] {cmd['icon']}  {cmd['command']}  ({'+'.join(cmd['keys'])})")
        return cmd

    # ── CONVENIENCE WRAPPERS (for UI button fallbacks) ────────────────────────────

    def simulate_cut(self):    self._fire([_MOD_KEY, "x"])
    def simulate_copy(self):   self._fire([_MOD_KEY, "c"])
    def simulate_paste(self):  self._fire([_MOD_KEY, "v"])
    def simulate_bold(self):   self._fire([_MOD_KEY, "b"])
    def simulate_italic(self): self._fire([_MOD_KEY, "i"])
    def simulate_undo(self):   self._fire([_MOD_KEY, "z"])

    # ── INTERNAL ──────────────────────────────────────────────────────────────────

    def _fire(self, keys: list[str]):
        if self.dry_run:
            print(f"  [dry-run] hotkeys({', '.join(keys)})")
            return
        try:
            pyautogui.hotkey(*keys)
        except Exception as exc:
            print(f"[CommandExecutor] hotkey failed: {exc}")


# ─── GESTURE METADATA (used by UI) ───────────────────────────────────────────────

def get_gesture_info() -> list[dict]:
    """Return sorted list of all gesture→command mappings for display."""
    return [
        {
            "gesture":     k,
            **v,
            "shortcut":    f"{'⌘' if _IS_MACOS else 'Ctrl'}+{v['keys'][-1].upper()}",
        }
        for k, v in COMMAND_MAP.items()
    ]
