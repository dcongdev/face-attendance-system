import tkinter as tk
from tkinter import ttk


class StatusIndicator(ttk.Frame):
    COLORS = {
        "idle": "#808080", "registering": "#eab308", "unknown": "#dc2626",
        "error": "#dc2626", "recognized": "#16a34a", "checked_in": "#16a34a",
        "registered": "#16a34a", "already_checked_in": "#2563eb",
    }

    def __init__(self, master):
        super().__init__(master, padding=10)
        self.light = tk.Canvas(self, width=20, height=20, highlightthickness=0)
        self.light.pack(side="left", padx=(0, 8))
        self._dot = self.light.create_oval(3, 3, 17, 17, outline="")
        self.label = ttk.Label(self, wraplength=740)
        self.label.pack(side="left", fill="x", expand=True)
        self._blink_id = None
        self._hold_id = None
        self._pending = None
        self._lit = True
        self.state = "idle"
        self.set("idle", "Ready")

    def set(self, state, message, *, hold_ms=0, force=False):
        """Hold event results briefly; coalesce background updates to the latest one."""
        if self._hold_id is not None and not force:
            self._pending = (state, message)
            return
        self.cancel_timers()
        self.state = state
        self._lit = True
        self.light.itemconfigure(self._dot, fill=self.COLORS[state])
        self.label.configure(text=f"Status: {message}")
        if state == "registering":
            self._blink_id = self.after(450, self._blink)
        if hold_ms:
            self._hold_id = self.after(hold_ms, self._release_hold)

    def _blink(self):
        self._blink_id = None
        self._lit = not self._lit
        self.light.itemconfigure(self._dot, fill=self.COLORS["registering" if self._lit else "idle"])
        self._blink_id = self.after(450, self._blink)

    def _release_hold(self):
        self._hold_id = None
        state, message = self._pending or ("idle", "Ready")
        self._pending = None
        self.set(state, message)

    def cancel_timers(self):
        for attribute in ("_blink_id", "_hold_id"):
            callback = getattr(self, attribute)
            if callback is not None:
                self.after_cancel(callback)
                setattr(self, attribute, None)
        self._pending = None

    def destroy(self):
        self.cancel_timers()
        super().destroy()
