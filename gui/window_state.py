import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys


STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "window_state.json"
DEFAULT_GEOMETRY = "1000x650"


def monitor_rectangles(window):
    """Return individual monitor bounds, including negative desktop coordinates."""
    if sys.platform == "win32":
        rectangles = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HANDLE, wintypes.HDC,
            ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
        )

        def collect(monitor, dc, rect, data):
            bounds = rect.contents
            rectangles.append((bounds.left, bounds.top, bounds.right, bounds.bottom))
            return True

        callback = callback_type(collect)
        if ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0):
            return rectangles
    return [(0, 0, window.winfo_screenwidth(), window.winfo_screenheight())]


def load_geometry(window, path=STATE_PATH):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        width, height, x, y = (state[key] for key in ("width", "height", "x", "y"))
        if any(type(value) is not int for value in (width, height, x, y)):
            return DEFAULT_GEOMETRY
        if not (800 <= width <= 32767 and 550 <= height <= 32767):
            return DEFAULT_GEOMETRY
        # Require a usable title-bar area on a real monitor, not a gap between them.
        visible = any(
            min(x + width, right) - max(x, left) >= 100
            and top <= y and y + 40 <= bottom
            for left, top, right, bottom in monitor_rectangles(window)
        )
        if visible:
            # Explicit '+' anchors positions to the left/top, even for negative values.
            return f"{width}x{height}+{x}+{y}"
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return DEFAULT_GEOMETRY


def save_geometry(window, path=STATE_PATH):
    # Minimized windows can report special off-screen system coordinates.
    if window.state() == "iconic":
        return
    state = {
        "width": window.winfo_width(), "height": window.winfo_height(),
        "x": window.winfo_x(), "y": window.winfo_y(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass  # An unwritable preferences file must not prevent camera cleanup or exit.
