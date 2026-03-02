import calendar
from datetime import date
import tkinter as tk
from tkinter import ttk


class DatePicker(tk.Toplevel):
    """Small calendar using only standard Tkinter/ttk widgets."""

    def __init__(self, master, selected, on_select):
        super().__init__(master)
        self.title("Select Date")
        self.resizable(False, False)
        self.transient(master)
        self.year, self.month = selected.year, selected.month
        self.on_select = on_select
        self.content = ttk.Frame(self, padding=10)
        self.content.pack()
        self._render()

    def _render(self):
        for widget in self.content.winfo_children():
            widget.destroy()
        ttk.Button(self.content, text="‹", width=3, command=lambda: self.move_month(-1)).grid(row=0, column=0)
        ttk.Label(self.content, text=f"{calendar.month_name[self.month]} {self.year}", anchor="center").grid(row=0, column=1, columnspan=5)
        ttk.Button(self.content, text="›", width=3, command=lambda: self.move_month(1)).grid(row=0, column=6)
        for column, day in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
            ttk.Label(self.content, text=day, anchor="center").grid(row=1, column=column)
        for row, week in enumerate(calendar.Calendar().monthdayscalendar(self.year, self.month), start=2):
            for column, day in enumerate(week):
                if day:
                    ttk.Button(self.content, text=str(day), width=4,
                               command=lambda day=day: self.select_day(day)).grid(row=row, column=column, padx=1, pady=1)

    def move_month(self, delta):
        year, month = divmod(self.year * 12 + self.month - 1 + delta, 12)
        if 1 <= year <= 9999:
            self.year, self.month = year, month + 1
            self._render()

    def select_day(self, day):
        self.on_select(date(self.year, self.month, day))
        self.destroy()
