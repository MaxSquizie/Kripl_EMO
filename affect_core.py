# EMO/affect_core.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional
import time
import math


@dataclass
class AffectState:
    # tonic mood (медленное, тянется к homeo)
    moodP: float = 0.10
    moodA: float = 0.00
    moodD: float = 0.00
    # burst (быстрое, затухает к нулю)
    burstP: float = 0.0
    burstA: float = 0.0
    burstD: float = 0.0
    # эффективное состояние для политики
    P_eff: float = 0.10
    A_eff: float = 0.00
    D_eff: float = 0.00
    # ДВЕ эмоции: по эффективному PAD и по чистому mood
    emotion_eff: str = "neutral"
    emotion_mood: str = "neutral"
    ts: float = 0.0


@dataclass
class AffectConfig:
    # база для mood
    homeo: Dict[str, float] = None

    # mood: инертность + deadband (мелочь игнорируем)
    decay_mood: Dict[str, float] = None            # скорость возврата mood → homeo (меньше = медленнее)
    gain_mood_from_event: float = 0.10             # сколько события уходит в mood (тонкий след)
    mood_deadband: Dict[str, float] = None         # |Δ| ниже порога — не влияет на mood

    # burst: быстрый и живой (чувствительный + нелинейное усиление)
    tau_burst: Dict[str, float] = None             # сек до e^-1 (распад к 0)
    gain_burst_P: float = 1.00
    gain_burst_A: float = 0.80
    gain_burst_D: float = 1.00
    burst_sens_gamma: float = 2.2                  # нелинейное усиление малых Δ (softplus)
    burst_min_gate: float = 0.04                   # микро-порог по |Δ| для гейта

    # вклад burst в эффективный PAD
    k_burst_P: float = 1.00
    k_burst_A: float = 0.95                        # чуть ярче вклад всплеска в A_eff
    k_burst_D: float = 1.00

    # пороги «бурности» (мягче для позитива)
    burst_thr_posP: float = 0.08
    burst_thr_negP: float = 0.22
    burst_thr_A: float = 0.18
    burst_thr_D: float = 0.16

    # доверие/интенсивность для гейта
    burst_conf_pos: float = 0.48
    burst_conf_neg: float = 0.58
    burst_inten: float = 0.10

    # гистерезис эмоций (подавление дрожания метки)
    hysteresis_A: float = 0.03
    hysteresis_P: float = 0.03

    # клип всего
    clip: float = 1.0

    def __post_init__(self):
        if self.homeo is None:
            self.homeo = {"P": 0.10, "A": 0.00, "D": 0.00}
        if self.decay_mood is None:
            # инертный mood: медленно тянется к базе
            self.decay_mood = {"P": 0.012, "A": 0.024, "D": 0.012}
        if self.mood_deadband is None:
            # мелкие события уходят только в burst, не в mood
            self.mood_deadband = {"P": 0.06, "A": 0.05, "D": 0.05}
        if self.tau_burst is None:
            # быстрый спад для «живости»
            self.tau_burst = {"P": 2.0, "A": 1.4, "D": 1.8}


class AffectEngine:
    """
    Двухслойная модель: mood (тoник) + burst (фазический).
    update(deltas, meta):
      - часть дельт идёт в burst (если событие «бурное»), тонкая доля — в mood (с deadband)
      - burst экспоненциально затухает к 0; mood тянется к homeo медленно
      - эффективный PAD: mood + k_burst_* * burst
      - эмоции считаются ОДИНАКОВОЙ картой: отдельно для eff и отдельно для mood
    """
    def __init__(
        self,
        cfg: Optional[AffectConfig] = None,
        logger: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.cfg = cfg or AffectConfig()
        self.state = AffectState(ts=time.time())
        self.log = logger
        # «прошлые» метки — для гистерезиса
        self._last_emotion_eff = "neutral"
        self._last_emotion_mood = "neutral"

    # ---------- утилиты ---------- #
    def _clip(self, x: float) -> float:
        c = self.cfg.clip
        return max(-c, min(c, x))

    # ---------- burst: затухает к 0 ---------- #
    def _decay_burst(self, dt: float):
        for k, tau in self.cfg.tau_burst.items():
            val = getattr(self.state, f"burst{k}")
            if tau <= 1e-3:
                newv = 0.0
            else:
                newv = val * math.exp(-dt / tau)
            setattr(self.state, f"burst{k}", self._clip(newv))

    def _inject_burst(self, dP: float, dA: float, dD: float):
        # нелинейное усиление мягких Δ: softplus(|x|*gamma)*sign(x)
        def spur(x: float) -> float:
            g = self.cfg.burst_sens_gamma
            s = 1.0 if x >= 0 else -1.0
            y = math.log1p(math.exp(g * abs(x))) / g  # softplus
            return s * y

        bP = spur(dP) * self.cfg.gain_burst_P
        bA = spur(dA) * self.cfg.gain_burst_A
        bD = spur(dD) * self.cfg.gain_burst_D

        self.state.burstP = self._clip(self.state.burstP + bP)
        self.state.burstA = self._clip(self.state.burstA + bA)
        self.state.burstD = self._clip(self.state.burstD + bD)

    # ---------- mood: медленный слой с homeostasis и deadband ---------- #
    def _update_mood(self, dP: float, dA: float, dD: float, dt: float):
        gm = self.cfg.gain_mood_from_event

        def after_deadband(x: float, thr: float) -> float:
            ax = abs(x)
            if ax <= thr:
                return 0.0
            # вычесть порог, сохранив знак — непрерывность перехода
            return (ax - thr) * (1.0 if x >= 0 else -1.0)

        for k, dX in [("P", dP), ("A", dA), ("D", dD)]:
            x = getattr(self.state, f"mood{k}")
            lam = self.cfg.decay_mood[k]
            base = self.cfg.homeo[k]
            thr = self.cfg.mood_deadband[k]
            dX_eff = after_deadband(dX, thr)  # мелочь — не влияет на mood
            nx = x + gm * dX_eff - lam * (x - base) * dt
            setattr(self.state, f"mood{k}", self._clip(nx))

    # ---------- combine ---------- #
    def _combine_effective(self):
        self.state.P_eff = self._clip(self.state.moodP + self.cfg.k_burst_P * self.state.burstP)
        self.state.A_eff = self._clip(self.state.moodA + self.cfg.k_burst_A * self.state.burstA)
        self.state.D_eff = self._clip(self.state.moodD + self.cfg.k_burst_D * self.state.burstD)

    # ---------- гейт «бурности» ---------- #
    def _is_bursty(self, dP: float, dA: float, dD: float, meta: Optional[Dict[str, Any]]) -> bool:
        inten = float(meta.get("intensity_raw", 0.0)) if meta else 0.0
        conf = 0.0
        if meta and isinstance(meta.get("probs"), dict):
            try:
                conf = float(max(meta["probs"].values()))
            except Exception:
                conf = 0.0
        s_cal = float(meta.get("sentiment_score_cal", 0.0)) if meta else 0.0  # [-1,1]

        # микро-порог по модулю Δ (не триггерим на крошечном шуме)
        min_gate = self.cfg.burst_min_gate
        has_energy = (abs(dP) > min_gate) or (abs(dA) > min_gate) or (abs(dD) > min_gate)

        # осевые условия
        pos_strong = (dP >= self.cfg.burst_thr_posP) or (s_cal >= 0.18)
        neg_strong = (dP <= -self.cfg.burst_thr_negP) or (s_cal <= -0.20)
        a_strong   = abs(dA) >= self.cfg.burst_thr_A
        d_strong   = abs(dD) >= self.cfg.burst_thr_D

        # доверие/интенсивность
        pos_gate = pos_strong and ((inten >= self.cfg.burst_inten) or (conf >= self.cfg.burst_conf_pos))
        neg_gate = neg_strong and ((inten >= self.cfg.burst_inten) or (conf >= self.cfg.burst_conf_neg))

        cond = has_energy and (pos_gate or neg_gate or a_strong or d_strong)

        # форс-триггеры (если передаются)
        if meta and isinstance(meta.get("flags"), dict):
            if any(meta["flags"].get(k, False) for k in ("urgent", "critique", "praise")):
                cond = True
        return cond

    # ---------- универсальная карта эмоций по PAD ---------- #
    def _classify_pad(self, P: float, A: float, D: float, prev: str) -> str:
        if P < -0.30 and A > 0.35 and D <= 0.15:
            emo = "fear"
        elif P < -0.30 and A > 0.35 and D > 0.15:
            emo = "anger"
        elif P < -0.35 and A < -0.10:
            emo = "sad"
        elif P > 0.30 and A <= 0.85:
            emo = "joy"
        elif A > 0.55 and P > -0.05:
            emo = "interest"
        elif -0.08 <= P <= 0.12 and -0.15 <= A <= 0.15:
            emo = "neutral"
        elif P > 0.12 and A < -0.20:
            emo = "calm"
        else:
            emo = "neutral"

        # мягкий гистерезис (подавляем мельтешение метки при малом сдвиге)
        dP = abs(P)
        dA = abs(A)
        if prev and prev != emo:
            if dP < self.cfg.hysteresis_P and dA < self.cfg.hysteresis_A:
                emo = prev
        return emo

    # ---------- публичный апдейт ---------- #
    def update(self, deltas: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> AffectState:
        now = time.time()
        dt = max(1e-3, now - self.state.ts)
        self.state.ts = now

        dP = float(deltas.get("dP", 0.0))
        dA = float(deltas.get("dA", 0.0))
        dD = float(deltas.get("dD", 0.0))

        # 1) распад быстрых всплесков
        self._decay_burst(dt)

        # 2) распил события: burst (если «бурно») + тонкая доля в mood (с deadband)
        if self._is_bursty(dP, dA, dD, meta):
            self._inject_burst(dP, dA, dD)
        self._update_mood(dP, dA, dD, dt)

        # 3) собрать эффективное состояние
        self._combine_effective()

        # 4) классифицировать по eff и по mood ОДНОЙ И ТОЙ ЖЕ картой
        self.state.emotion_eff = self._classify_pad(
            self.state.P_eff, self.state.A_eff, self.state.D_eff, self._last_emotion_eff
        )
        self.state.emotion_mood = self._classify_pad(
            self.state.moodP, self.state.moodA, self.state.moodD, self._last_emotion_mood
        )

        # обновить гистерезисные «предыдущие»
        self._last_emotion_eff = self.state.emotion_eff
        self._last_emotion_mood = self.state.emotion_mood

        if self.log:
            self.log("affect_tick", {
                "dt": dt,
                "deltas": {"dP": dP, "dA": dA, "dD": dD},
                "meta": meta or {},
                "state": asdict(self.state)
            })
        return self.state

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self.state)

    def reward(self, kind: str, value: float, source: str = "system", attach: Optional[Dict[str, Any]] = None):
        if self.log:
            self.log("reward", {"kind": kind, "value": value, "source": source, "attach": attach or {}})
