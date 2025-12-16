# -*- coding: utf-8 -*-
import os
import time
import tkinter as tk
from tkinter import ttk, scrolledtext

EMO_CONFIG_PATH = r"C:\Users\Ilya\PycharmProjects\Cripl\Cripl\EMO\emo_config.json"

from kripl_affect import KriplAffect


class AffectGUI(tk.Tk):
    def __init__(self, config_path: str):
        super().__init__()
        self.title("Kripl EMO — realtime mood / burst / eff (v3)")
        self.geometry("1120x720")
        self.minsize(1040, 620)

        # движок
        self.engine = KriplAffect(config_path)
        self.last_tick = time.time()
        self.tick_interval_ms = 200  # важно: до UI

        # UI
        self._build_layout()
        self._style_numbers(self)

        # таймер
        self.after(self.tick_interval_ms, self._on_timer)

        # первый снимок
        self._append_snapshot_to_console(prefix="INIT")
        self._apply_snapshot_to_vars(self.engine.state)

    # ---------- UI ----------
    def _build_layout(self):
        top = ttk.Frame(self, padding=(10, 10, 10, 5))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="User prompt →", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.entry = ttk.Entry(top, font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.entry.bind("<Return>", self.on_send)

        ttk.Button(top, text="Send", command=self.on_send).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear", command=self._clear_console).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top, text="  Tick (ms):").pack(side=tk.LEFT, padx=(12, 4))
        self.spin_tick = ttk.Spinbox(top, from_=50, to=2000, increment=50, width=6)
        self.spin_tick.set(str(self.tick_interval_ms))
        self.spin_tick.pack(side=tk.LEFT)

        sep = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=10, pady=5)

        center = ttk.Frame(self, padding=(10, 0, 10, 10))
        center.pack(fill=tk.BOTH, expand=True)

        # left: log
        left = ttk.Frame(center)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Log", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.console = scrolledtext.ScrolledText(left, height=20, font=("Consolas", 10))
        self.console.pack(fill=tk.BOTH, expand=True)

        # right: stats
        right = ttk.Frame(center)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))

        # USER
        pane_user = self._group(right, "USER (perception)")
        self.var_user_label = tk.StringVar(value="neutral")
        self.var_user_prob = tk.StringVar(value="0.00")
        self.var_user_entropy = tk.StringVar(value="1.00")
        self.var_user_pad = tk.StringVar(value="( +0.00, +0.00, +0.00 )")
        self._kv(pane_user, "top_label:", self.var_user_label)
        self._kv(pane_user, "top_prob:", self.var_user_prob)
        self._kv(pane_user, "entropy_norm:", self.var_user_entropy)
        self._kv(pane_user, "PAD_user (P,A,D):", self.var_user_pad)

        # BURST
        pane_burst = self._group(right, "KRIPL / burst (всплеск)")
        self.var_burst_label = tk.StringVar(value="neutral")
        self.var_burst_pad = tk.StringVar(value="( +0.00, +0.00, +0.00 )")
        self.var_mark_burst = tk.StringVar(value="0.00")
        self.var_mark_impulse = tk.StringVar(value="0.00")
        self.var_hold_left = tk.StringVar(value="0.00 s")
        self._kv(pane_burst, "label:", self.var_burst_label)
        self._kv(pane_burst, "PAD:", self.var_burst_pad)
        self._kv(pane_burst, "mark.burst:", self.var_mark_burst)
        self._kv(pane_burst, "mark.impulse:", self.var_mark_impulse)
        self._kv(pane_burst, "hold_left:", self.var_hold_left)

        # MOOD
        pane_mood = self._group(right, "KRIPL / mood (фон)")
        self.var_mood_label = tk.StringVar(value="neutral")
        self.var_mood_pad = tk.StringVar(value="( +0.00, +0.00, +0.00 )")
        self._kv(pane_mood, "label:", self.var_mood_label)
        self._kv(pane_mood, "PAD:", self.var_mood_pad)

        # EFF
        pane_eff = self._group(right, "KRIPL / eff (выражение)")
        self.var_eff_label = tk.StringVar(value="neutral")
        self.var_eff_pad = tk.StringVar(value="( +0.00, +0.00, +0.00 )")
        self._kv(pane_eff, "label:", self.var_eff_label)
        self._kv(pane_eff, "PAD:", self.var_eff_pad)

        # META
        pane_meta = self._group(right, "META")
        self.var_beta = tk.StringVar(value="0.00")
        self.var_fatigue = tk.StringVar(value="0.00")
        self.var_domain = tk.StringVar(value="meta")
        self.var_ctx = tk.StringVar(value="1.00")
        self.var_len = tk.StringVar(value="0.00")
        self._kv(pane_meta, "β (burst weight):", self.var_beta)
        self._kv(pane_meta, "fatigue:", self.var_fatigue)
        self._kv(pane_meta, "domain:", self.var_domain)
        self._kv(pane_meta, "context_factor:", self.var_ctx)
        self._kv(pane_meta, "len_norm:", self.var_len)

        # status bar
        self.status = ttk.Label(self, anchor="w", relief=tk.SUNKEN, padding=(8, 2))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._update_status("ready.")

    def _group(self, parent, title):
        frm = ttk.LabelFrame(parent, text=title, padding=(8, 6, 8, 8))
        frm.pack(fill=tk.X, anchor="n", pady=(0, 8))
        return frm

    def _kv(self, parent, key, var: tk.StringVar):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=key, width=18, anchor="w").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, anchor="w", font=("Consolas", 10)).pack(side=tk.LEFT)

    def _style_numbers(self, root):
        try:
            s = ttk.Style(root)
            if "vista" in s.theme_names():
                s.theme_use("vista")
            elif "clam" in s.theme_names():
                s.theme_use("clam")
        except Exception:
            pass

    # ---------- events ----------
    def on_send(self, event=None):
        try:
            self.tick_interval_ms = int(self.spin_tick.get())
        except Exception:
            pass

        txt = self.entry.get().strip()
        if not txt:
            return

        st = self.engine.step(txt)

        self._append(f"[USER PROMPT] {txt}\n")
        self._append_snapshot_to_console()
        self._apply_snapshot_to_vars(st)
        self.entry.delete(0, tk.END)

    # ---------- timer ----------
    def _on_timer(self):
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now

        self.engine.idle_tick(dt)
        st = self.engine.state
        self._apply_snapshot_to_vars(st)

        self._update_status(
            f"dt={dt:.2f}s | β={st.beta_current:.2f} | marks: burst={st.burst_mark:.2f}, impulse={st.impulse_mark:.2f}, hold_left={st.hold_left_s:.2f}s"
        )
        self.after(self.tick_interval_ms, self._on_timer)

    # ---------- helpers ----------
    def _fmt_pad(self, p, a, d):
        return f"( {p:+.2f}, {a:+.2f}, {d:+.2f} )"

    def _apply_snapshot_to_vars(self, st):
        # USER
        self.var_user_label.set(st.user_top)
        self.var_user_prob.set(f"{st.user_top_prob:.2f}")
        self.var_user_entropy.set(f"{st.user_entropy:.2f}")
        self.var_user_pad.set(self._fmt_pad(*st.user_pad))

        # BURST
        self.var_burst_label.set(st.burst_label)
        self.var_burst_pad.set(self._fmt_pad(*st.burst))
        self.var_mark_burst.set(f"{st.burst_mark:.2f}")
        self.var_mark_impulse.set(f"{st.impulse_mark:.2f}")
        self.var_hold_left.set(f"{st.hold_left_s:.2f} s")

        # MOOD
        self.var_mood_label.set(st.mood_label)
        self.var_mood_pad.set(self._fmt_pad(*st.mood))

        # EFF
        self.var_eff_label.set(st.eff_label)
        self.var_eff_pad.set(self._fmt_pad(*st.eff))

        # META
        self.var_beta.set(f"{st.beta_current:.2f}")
        self.var_fatigue.set(f"{st.fatigue:.2f}")
        self.var_domain.set(st.domain)
        self.var_ctx.set(f"{st.context_factor:.2f}")
        self.var_len.set(f"{st.len_norm:.2f}")

    def _append_snapshot_to_console(self, prefix: str | None = None):
        snap = self.engine.snapshot()
        U = snap["USER"]
        K = snap["KRIPL"]
        marks = K.get("marks", {})

        lines = []
        if prefix:
            lines.append(f"=== {prefix} SNAPSHOT ===")
        lines.append(
            "USER: top={top} prob={prob:.2f} Hn={H:.2f} pad=( {p:+.2f}, {a:+.2f}, {d:+.2f} )".format(
                top=U["top_label"], prob=U["top_prob"], H=U["entropy_norm"],
                p=U["pad_user"]["P"], a=U["pad_user"]["A"], d=U["pad_user"]["D"]
            )
        )
        lines.append(
            "BURST: label={lbl} pad=( {p:+.2f}, {a:+.2f}, {d:+.2f} )  marks: burst={bm:.2f} impulse={im:.2f} hold_left={hl:.2f}s".format(
                lbl=K["burst"]["label"],
                p=K["burst"]["P"], a=K["burst"]["A"], d=K["burst"]["D"],
                bm=marks.get("burst", 0.0), im=marks.get("impulse", 0.0), hl=marks.get("hold_left_s", 0.0)
            )
        )
        lines.append(
            "MOOD:  label={lbl} pad=( {p:+.2f}, {a:+.2f}, {d:+.2f} )".format(
                lbl=K["mood"]["label"], p=K["mood"]["P"], a=K["mood"]["A"], d=K["mood"]["D"]
            )
        )
        lines.append(
            "EFF:   label={lbl} pad=( {p:+.2f}, {a:+.2f}, {d:+.2f} )".format(
                lbl=K["eff"]["label"], p=K["eff"]["P"], a=K["eff"]["A"], d=K["eff"]["D"]
            )
        )
        lines.append(
            "META:  β={b:.2f} fatigue={f:.2f} domain={d} ctx={c:.2f} len_norm={ln:.2f}".format(
                b=marks.get("beta_current", 0.0),
                f=marks.get("fatigue", 0.0),
                d=marks.get("domain", "meta"),
                c=marks.get("context_factor", 1.0),
                ln=marks.get("len_norm", 0.0)
            )
        )
        lines.append("")
        self._append("\n".join(lines))

    def _append(self, text: str):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def _clear_console(self):
        self.console.delete("1.0", tk.END)

    def _update_status(self, text: str):
        self.status.config(text=text)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = AffectGUI(EMO_CONFIG_PATH)
    app.mainloop()


if __name__ == "__main__":
    main()
