import tkinter as tk
from tkinter import font as tkfont

from PIL import Image, ImageDraw

from src.inference import load_digit_model, predict_from_image
from src.paths import APP_ICON_PNG_PATH

THEME = {
    "bg": "#08111E",
    "surface": "#112033",
    "surface_alt": "#16293F",
    "surface_hover": "#19304B",
    "border": "#27415F",
    "text": "#EDF4FA",
    "text_muted": "#8FA4B9",
    "primary": "#FF7A3D",
    "primary_hover": "#FF915B",
    "accent": "#47D6C6",
    "canvas_border": "#D7E4EE",
    "canvas_bg": "#FFFFFF",
    "danger": "#FF7A7A",
}

BAR_COLORS = {"default": THEME["primary"], "top": THEME["accent"]}


class App(tk.Tk):
    CANVAS_SIZE = 280
    BRUSH_RADIUS = 8

    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.configure(bg=THEME["bg"])
        self.geometry("960x680")
        self.minsize(900, 640)
        self._icon_image = None
        self._logo_image = None
        self._bar_render_job = None
        self._current_probs = None
        self._set_window_icon()
        self._init_fonts()
        self.model = load_digit_model()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        shell = tk.Frame(self, bg=THEME["bg"], padx=24, pady=20)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, minsize=320)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg=THEME["bg"])
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        self._build_header(header)

        left_card = tk.Frame(
            shell,
            bg=THEME["surface"],
            bd=1,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            padx=18,
            pady=18,
        )
        left_card.grid(row=1, column=0, sticky="nsw", padx=(0, 18))

        tk.Label(
            left_card,
            text="手写画布",
            font=self.fonts["section_title"],
            bg=THEME["surface"],
            fg=THEME["text"],
        ).pack(anchor="w")
        tk.Label(
            left_card,
            text="在画布中写下 0-9 数字",
            font=self.fonts["caption"],
            bg=THEME["surface"],
            fg=THEME["text_muted"],
        ).pack(anchor="w", pady=(4, 14))

        canvas_shell = tk.Frame(
            left_card,
            bg=THEME["canvas_bg"],
            highlightthickness=1,
            highlightbackground=THEME["canvas_border"],
            bd=0,
            padx=10,
            pady=10,
        )
        canvas_shell.pack()

        self.canvas = tk.Canvas(
            canvas_shell,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg=THEME["canvas_bg"],
            cursor="cross",
            highlightthickness=0,
        )
        self.canvas.pack()

        buttons = tk.Frame(left_card, bg=THEME["surface"])
        buttons.pack(fill=tk.X, pady=(16, 0))
        self.recognize_button = self._create_button(
            buttons,
            text="识别",
            command=self._classify,
            background=THEME["primary"],
            active_background=THEME["primary_hover"],
            foreground="#FFFFFF",
        )
        self.recognize_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.clear_button = self._create_button(
            buttons,
            text="清除",
            command=self._clear,
            background=THEME["surface"],
            active_background=THEME["surface_hover"],
            foreground=THEME["text"],
            border_color=THEME["border"],
        )
        self.clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        right = tk.Frame(shell, bg=THEME["bg"])
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        result_card = tk.Frame(
            right,
            bg=THEME["surface"],
            bd=1,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            padx=20,
            pady=20,
        )
        result_card.grid(row=0, column=0, sticky="ew")

        tk.Label(
            result_card,
            text="预测结果",
            font=self.fonts["label"],
            bg=THEME["surface"],
            fg=THEME["text_muted"],
        ).pack(anchor="w")

        result_frame = tk.Frame(result_card, bg=THEME["surface"])
        result_frame.pack(fill=tk.X, pady=(10, 0))

        self.label_digit = tk.Label(
            result_frame,
            text="?",
            font=self.fonts["digit"],
            fg=THEME["accent"],
            bg=THEME["surface"],
            width=2,
        )
        self.label_digit.pack(side=tk.LEFT, padx=(0, 14))

        conf_frame = tk.Frame(result_frame, bg=THEME["surface"])
        conf_frame.pack(side=tk.LEFT, anchor="w")
        tk.Label(
            conf_frame,
            text="LeNet-5 推理结果",
            font=self.fonts["section_title"],
            bg=THEME["surface"],
            fg=THEME["text"],
        ).pack(anchor="w")
        self.label_conf = tk.Label(
            conf_frame,
            text="书写数字后点击识别",
            font=self.fonts["body"],
            bg=THEME["surface"],
            fg=THEME["text_muted"],
        )
        self.label_conf.pack(anchor="w", pady=(6, 0))

        bars_card = tk.Frame(
            right,
            bg=THEME["surface"],
            bd=1,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            padx=20,
            pady=18,
        )
        bars_card.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        bars_card.columnconfigure(0, weight=1)

        tk.Label(
            bars_card,
            text="分类概率",
            font=self.fonts["section_title"],
            bg=THEME["surface"],
            fg=THEME["text"],
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            bars_card,
            text="高亮条为最可能的数字",
            font=self.fonts["caption"],
            bg=THEME["surface"],
            fg=THEME["text_muted"],
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        bars_frame = tk.Frame(bars_card, bg=THEME["surface"])
        bars_frame.grid(row=2, column=0, sticky="nsew")

        self._bar_canvases = []
        self._bar_labels = []
        for i in range(10):
            row = tk.Frame(bars_frame, bg=THEME["surface"])
            row.pack(fill=tk.X, pady=2)

            lbl = tk.Label(
                row,
                text=str(i),
                font=self.fonts["bar_label"],
                width=2,
                anchor="e",
                bg=THEME["surface"],
                fg=THEME["text_muted"],
            )
            lbl.pack(side=tk.LEFT)

            bar_cv = tk.Canvas(
                row,
                height=18,
                bg=THEME["bg"],
                highlightthickness=0,
                bd=0,
            )
            bar_cv.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
            bar_cv.bind("<Configure>", self._schedule_bar_render)

            val = tk.Label(
                row,
                text="0%",
                font=self.fonts["caption"],
                width=6,
                anchor="e",
                bg=THEME["surface"],
                fg=THEME["text_muted"],
            )
            val.pack(side=tk.LEFT)

            self._bar_canvases.append(bar_cv)
            self._bar_labels.append(val)

        footer = tk.Label(
            shell,
            text="FML-Project · Python GUI 桌面版",
            font=self.fonts["caption"],
            bg=THEME["bg"],
            fg=THEME["text_muted"],
        )
        footer.grid(row=2, column=0, columnspan=2, pady=(16, 0))

        self._pil_image = Image.new(
            "RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), THEME["canvas_bg"]
        )
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self._last_x = None
        self._last_y = None

        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._reset_pos)
        self.bind("<Return>", lambda _event: self._classify())
        self.bind("<Escape>", lambda _event: self._clear())

        self._render_bars(None)

    def _init_fonts(self):
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.fonts = {
            "title": (family, 24, "bold"),
            "subtitle": (family, 11),
            "section_title": (family, 13, "bold"),
            "label": (family, 11, "bold"),
            "body": (family, 12),
            "caption": (family, 10),
            "digit": (family, 56, "bold"),
            "button": (family, 12, "bold"),
            "bar_label": (family, 11, "bold"),
        }

    def _set_window_icon(self):
        if not APP_ICON_PNG_PATH.exists():
            return
        try:
            self._icon_image = tk.PhotoImage(file=str(APP_ICON_PNG_PATH))
            self.iconphoto(True, self._icon_image)
        except tk.TclError:
            self._icon_image = None

    def _build_header(self, parent):
        brand = tk.Frame(parent, bg=THEME["bg"])
        brand.pack(anchor="w", fill=tk.X)

        if APP_ICON_PNG_PATH.exists():
            try:
                self._logo_image = tk.PhotoImage(file=str(APP_ICON_PNG_PATH)).subsample(8, 8)
                tk.Label(
                    brand,
                    image=self._logo_image,
                    bg=THEME["bg"],
                ).pack(side=tk.LEFT, padx=(0, 14))
            except tk.TclError:
                self._logo_image = None

        copy = tk.Frame(brand, bg=THEME["bg"])
        copy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            copy,
            text="手写数字识别",
            font=self.fonts["title"],
            bg=THEME["bg"],
            fg=THEME["text"],
        ).pack(anchor="w")
        tk.Label(
            copy,
            text="Python GUI 版本",
            font=self.fonts["subtitle"],
            bg=THEME["bg"],
            fg=THEME["text_muted"],
        ).pack(anchor="w", pady=(4, 0))

    def _create_button(
        self,
        parent,
        *,
        text,
        command,
        background,
        active_background,
        foreground,
        border_color=None,
    ):
        border_bg = border_color or background
        outer = tk.Frame(parent, bg=border_bg, padx=1, pady=1)
        inner = tk.Label(
            outer,
            text=text,
            font=self.fonts["button"],
            bg=background,
            fg=foreground,
            cursor="hand2",
            padx=18,
            pady=10,
        )
        inner.pack(fill=tk.BOTH, expand=True)

        def on_enter(_e):
            inner.configure(bg=active_background)

        def on_leave(_e):
            inner.configure(bg=background)

        def on_click(_e):
            command()

        inner.bind("<Enter>", on_enter)
        inner.bind("<Leave>", on_leave)
        inner.bind("<Button-1>", on_click)

        return outer

    def _schedule_bar_render(self, _event=None):
        if self._bar_render_job is not None:
            self.after_cancel(self._bar_render_job)
        self._bar_render_job = self.after(20, self._refresh_bars)

    def _refresh_bars(self):
        self._bar_render_job = None
        self._render_bars(self._current_probs)

    def _render_bars(self, probs=None):
        self._current_probs = probs
        top_idx = -1
        if probs is not None:
            top_idx = max(range(10), key=lambda i: probs[i])

        for i in range(10):
            cv = self._bar_canvases[i]
            cv.delete("all")
            cv.update_idletasks()
            w = max(cv.winfo_width(), 1)
            pct = probs[i] if probs is not None else 0.0
            bar_w = max(0, int(w * pct))
            color = BAR_COLORS["top"] if i == top_idx else BAR_COLORS["default"]
            if bar_w > 0:
                cv.create_rectangle(0, 0, bar_w, 18, fill=color, outline="")
            self._bar_labels[i].configure(
                text=f"{pct * 100:.1f}%" if probs is not None else "0%"
            )

    def _paint_point(self, x, y):
        r = self.BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self._pil_draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def _start_draw(self, event):
        self._paint_point(event.x, event.y)
        self._last_x = event.x
        self._last_y = event.y

    def _draw(self, event):
        x, y = event.x, event.y
        if self._last_x is not None:
            self._pil_draw.line(
                [(self._last_x, self._last_y), (x, y)],
                fill="black", width=self.BRUSH_RADIUS * 2,
            )
            self.canvas.create_line(
                self._last_x,
                self._last_y,
                x,
                y,
                fill="black",
                width=self.BRUSH_RADIUS * 2,
                capstyle=tk.ROUND,
                smooth=True,
            )
        self._paint_point(x, y)
        self._last_x, self._last_y = x, y

    def _reset_pos(self, _event):
        self._last_x = self._last_y = None

    def _classify(self):
        if self.model is None:
            self.label_conf.configure(text="模型未加载", fg=THEME["danger"])
            return

        digit, confidence, probs = predict_from_image(self.model, self._pil_image)
        if digit is None:
            self.label_digit.configure(text="?")
            self.label_conf.configure(text="请先书写数字", fg=THEME["danger"])
            self._render_bars(None)
            return

        self.label_digit.configure(text=str(digit))
        self.label_conf.configure(
            text=f"置信度: {confidence * 100:.1f}%",
            fg=THEME["text_muted"],
        )
        self._render_bars(probs)

    def _clear(self):
        self.canvas.delete("all")
        self._pil_image = Image.new(
            "RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), THEME["canvas_bg"]
        )
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self.label_digit.configure(text="?")
        self.label_conf.configure(text="书写数字后点击识别", fg=THEME["text_muted"])
        self._render_bars(None)
        self._reset_pos(None)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
