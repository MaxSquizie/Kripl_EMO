# -*- coding: utf-8 -*-
"""
Kripl Affect v3
- Local-only backend (GoEmotionsRU) from local_model_dir
- TOP-k temp-mix for pad_user anchor; non-linear confidence & soft neutralization
- Burst priority: hold-window + post-hold tail (tracked by elapsed time)
- EMA burst_mark; refractory for frequent short bursts
- Mood is inertial; eff blends burst→mood over time with explicit β
- Labeling by Mahalanobis over multi-centroids (+quadrant penalties) with hysteresis
- PAD calibration (A,b); domain/context bias; length normalization; fatigue; micro-noise
- Snapshot exposes beta_current, fatigue, domain, context_factor, len_norm
"""

from __future__ import annotations
import os, json, math, time, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

# Offline env
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from emo_backend_goemotions import GoEmotionsRU, PAD_PROTOS_DEFAULT  # fallback protos


# -------------------- utils --------------------

def _entropy_norm(probs: List[float]) -> float:
    eps = 1e-12
    H = -sum(p * math.log(max(p, eps)) for p in probs)
    Hmax = math.log(len(probs) if probs else 1)
    return H / max(Hmax, 1e-12)

def _clip(x: float, limit: float) -> float:
    return max(-limit, min(limit, x))

def _path_norm(p: str) -> str:
    return os.path.normpath(os.path.expandvars(os.path.expanduser(p)))

def _norm3(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _exp_decay_factor_half_life(dt_s: float, half_life_s: float) -> float:
    if half_life_s <= 0:
        return 0.0
    tau = half_life_s / math.log(2.0)
    return math.exp(-max(0.0, dt_s) / tau)

def _exp_rise_factor_tau(dt_s: float, tau_s: float) -> float:
    if tau_s <= 0:
        return 1.0
    return 1.0 - math.exp(-max(0.0, dt_s) / tau_s)

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _len_norm(text: str, k: float = 90.0) -> float:
    # 0..1 по длине текста (логарифм)
    return max(0.0, min(1.0, math.log(1 + len(text)) / math.log(1 + k)))

def _ema(prev: float, target: float, alpha: float) -> float:
    return prev + alpha * (target - prev)


# -------------------- config dataclasses --------------------

@dataclass
class BackendCfg:
    device: Optional[str]
    min_top_prob: float
    entropy_neutral: float
    temp_mix: float
    neutral_weight: float
    use_top_only: bool
    topk: int

@dataclass
class PersonaCfg:
    baseline_P: float
    baseline_A: float
    baseline_D: float

@dataclass
class EmpathyCfg:
    W_empathy: float
    E: Dict[str, float]
    mood_follow_user: float

@dataclass
class TimeCfg:
    burst_half_life_s: float
    burst_rise_tau_s: float
    mood_decay_half_life_s: float
    mood_rise_tau_s: float
    beta_base: float
    beta_entropy_gain: float
    beta_burst_gain: float
    beta_burst_half_life_s: float
    beta_base_idle: float
    hold_seconds: float
    hold_beta: float
    post_hold_tau_s: float
    beta_impulse_gain: float
    beta_impulse_half_life_s: float
    refractory_s: float
    refractory_hold_scale: float
    refractory_beta_scale: float

@dataclass
class LayersCfg:
    burst_clip: float
    mood_clip: float
    pad_bounds: float

@dataclass
class LimitsCfg:
    pad_clip_per_step: float
    min_pad_norm_for_label: float

@dataclass
class LabelingCfg:
    mode: str
    min_pad_norm: float
    hysteresis: float


# -------------------- runtime state --------------------

@dataclass
class KriplAffectState:
    # USER
    user_text: str = ""
    user_top: str = "neutral"
    user_top_prob: float = 0.0
    user_entropy: float = 1.0
    user_pad: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # KRIPL layers
    mood: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    burst: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    eff: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # labels
    mood_label: str = "neutral"
    burst_label: str = "neutral"
    eff_label: str = "neutral"

    # marks / meta
    burst_mark: float = 0.0
    impulse_mark: float = 0.0
    hold_left_s: float = 0.0
    post_hold_elapsed_s: float = 0.0
    beta_current: float = 0.0
    fatigue: float = 0.0
    domain: str = "meta"
    context_factor: float = 1.0
    len_norm: float = 0.0

    # time
    ts: float = 0.0
    last_ts: float = 0.0
    last_hold_end_ts: float = 0.0

    # slow evidence integrator for mood
    mood_ev: Tuple[float, float, float] = (0.0, 0.0, 0.0)



# -------------------- engine --------------------

class KriplAffect:
    def __init__(self, config_path: str) -> None:
        self.cfg_raw = self._load_json(config_path)
        for key in ("paths", "backend", "empathy", "layers", "limits", "time", "persona", "labeling"):
            if key not in self.cfg_raw:
                raise KeyError(f"emo_config.json: missing '{key}' section")

        # paths
        self.paths = dict(self.cfg_raw["paths"])
        self.paths["local_model_dir"] = _path_norm(self.paths["local_model_dir"])
        self.paths["log_dir"] = _path_norm(self.paths.get("log_dir", "./logs"))

        # backend
        b = self.cfg_raw["backend"]
        self.backend_cfg = BackendCfg(
            device=b.get("device"),
            min_top_prob=float(b.get("min_top_prob", 0.40)),
            entropy_neutral=float(b.get("entropy_neutral", 0.96)),
            temp_mix=float(b.get("temp_mix", 1.30)),
            neutral_weight=float(b.get("neutral_weight", 0.20)),
            use_top_only=bool(b.get("use_top_only", False)),
            topk=int(b.get("topk", 5))
        )

        # persona
        p = self.cfg_raw["persona"]["baseline_PAD"]
        self.persona = PersonaCfg(
            baseline_P=float(p.get("P", 0.15)),
            baseline_A=float(p.get("A", 0.05)),
            baseline_D=float(p.get("D", 0.10)),
        )

        # empathy
        e = self.cfg_raw["empathy"]
        self.empathy = EmpathyCfg(
            W_empathy=float(e.get("W_empathy", 0.7)),
            E=e.get("E", {}),
            mood_follow_user=float(e.get("mood_follow_user", 0.10))
        )

        # sensitivity
        self.burst_gain = float(e.get("burst_gain", 1.0))
        self.mood_gain  = float(e.get("mood_gain", 1.0))

        # mood evidence integrator cfg
        me = self.cfg_raw.get("mood_evidence", {})
        self.mood_ev_gain = float(me.get("gain", 0.25))
        self.mood_ev_hl   = float(me.get("half_life_s", 180.0))
        self.mood_ev_clip = float(me.get("clip", 0.30))
        self.mood_ev_ctx  = float(me.get("context_scale", 0.6))


        # time
        t = self.cfg_raw["time"]
        self.time_cfg = TimeCfg(
            burst_half_life_s=float(t.get("burst_half_life_s", 12.0)),
            burst_rise_tau_s=float(t.get("burst_rise_tau_s", 0.35)),
            mood_decay_half_life_s=float(t.get("mood_decay_half_life_s", 2400.0)),
            mood_rise_tau_s=float(t.get("mood_rise_tau_s", 75.0)),
            beta_base=float(t.get("beta_base", 0.42)),
            beta_entropy_gain=float(t.get("beta_entropy_gain", 0.45)),
            beta_burst_gain=float(t.get("beta_burst_gain", 0.48)),
            beta_burst_half_life_s=float(t.get("beta_burst_half_life_s", 4.0)),
            beta_base_idle=float(t.get("beta_base_idle", 0.08)),
            hold_seconds=float(t.get("hold_seconds", 2.5)),
            hold_beta=float(t.get("hold_beta", 0.90)),
            post_hold_tau_s=float(t.get("post_hold_tau_s", 3.5)),
            beta_impulse_gain=float(t.get("beta_impulse_gain", 0.30)),
            beta_impulse_half_life_s=float(t.get("beta_impulse_half_life_s", 1.5)),
            refractory_s=float(t.get("refractory_s", 4.0)),
            refractory_hold_scale=float(t.get("refractory_hold_scale", 0.70)),
            refractory_beta_scale=float(t.get("refractory_beta_scale", 0.80))
        )

        # layers/limits/labeling
        l = self.cfg_raw["layers"]
        self.layers = LayersCfg(
            burst_clip=float(l.get("burst_clip", 0.85)),
            mood_clip=float(l.get("mood_clip", 0.50)),
            pad_bounds=float(l.get("pad_bounds", 1.0))
        )
        lim = self.cfg_raw["limits"]
        self.limits = LimitsCfg(
            pad_clip_per_step=float(lim.get("pad_clip_per_step", 0.60)),
            min_pad_norm_for_label=float(lim.get("min_pad_norm_for_label", 0.08))
        )
        lab = self.cfg_raw["labeling"]
        self.labeling = LabelingCfg(
            mode=str(lab.get("mode", "mahal")),
            min_pad_norm=float(lab.get("min_pad_norm", 0.08)),
            hysteresis=float(lab.get("hysteresis", 0.07))
        )

        # calibration & realism
        self.calibA = self.cfg_raw.get("calibration", {}).get("A", [[1,0,0],[0,1,0],[0,0,1]])
        self.calibb = self.cfg_raw.get("calibration", {}).get("b", [0,0,0])
        self.realism = self.cfg_raw.get("realism", {})
        self.domain_bias = self.realism.get("domain_bias", {})
        self.micro_noise_std = float(self.realism.get("micro_noise_std", 0.015))
        self.fatigue_cfg = self.realism.get("fatigue", {"threshold_A":0.4,"gain_per_sec":0.015,"relax_half_life_s":900.0,"impact":{"A":-0.3,"P":-0.05,"D":-0.1}})
        self.context_factor_base = float(self.realism.get("context_factor_base", 0.85))

        # multi-centroids for labeling
        self.pad_multi = self.cfg_raw.get("pad_protos", {})
        self.use_multi = bool(self.pad_multi)

        # backend (локально)
        local_dir = self.paths["local_model_dir"]
        device = self.backend_cfg.device
        os.environ.setdefault("CRIPL_EMO_LOCALDIR", local_dir)
        if device:
            os.environ.setdefault("CRIPL_EMO_DEVICE", device)
        self.backend = GoEmotionsRU(local_dir=local_dir, device=device)
        self.labels = self.backend.order

        # state
        now = time.time()
        base = (self.persona.baseline_P, self.persona.baseline_A, self.persona.baseline_D)
        self.state = KriplAffectState(
            mood=base,
            eff=base,
            ts=now,
            last_ts=now,
            last_hold_end_ts=now
        )

        try: os.makedirs(self.paths["log_dir"], exist_ok=True)
        except Exception: pass

    # ---------------- JSON ----------------
    def _load_json(self, path: str) -> Dict[str, Any]:
        p = _path_norm(path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------- PAD helpers ----------------
    def _pad_calibrate(self, pad):
        A = self.calibA; b = self.calibb
        P = A[0][0]*pad[0] + A[0][1]*pad[1] + A[0][2]*pad[2] + b[0]
        A_ = A[1][0]*pad[0] + A[1][1]*pad[1] + A[1][2]*pad[2] + b[1]
        D = A[2][0]*pad[0] + A[2][1]*pad[1] + A[2][2]*pad[2] + b[2]
        return (P, A_, D)

    def _pad_from_mix(self, probs: Dict[str, float]) -> Tuple[float, float, float]:
        """TOP-k температурная смесь прототипов (как якорь от восприятия)."""
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:max(1, self.backend_cfg.topk)]
        # температурная корректировка
        T = max(1e-6, self.backend_cfg.temp_mix)
        w = []
        for lbl, p in items:
            v = p if lbl != "neutral" else p * self.backend_cfg.neutral_weight
            w.append(v ** (1.0 / T))
        s = sum(w) + 1e-12
        w = [x / s for x in w]
        P = A = D = 0.0
        for (lbl, _), wi in zip(items, w):
            p,a,d = PAD_PROTOS_DEFAULT.get(lbl, (0.0,0.0,0.0))
            P += wi * p; A += wi * a; D += wi * d
        return (P, A, D)

    # ---------------- Domain / context ----------------
    def _detect_domain(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ("ошиб", "падение", "баг", "срочно", "дедлайн", "deadline", "сделай", "правь", "фикс")):
            return "task"
        if any(k in t for k in ("спасибо", "класс", "обожаю", "ты лучшая", "друж", "подруга")):
            return "social"
        return "meta"

    def _context_factor(self, text: str) -> float:
        base = self.context_factor_base
        t = text.lower()
        bonus = 0.0
        if "@" in t or "крипл" in t:
            bonus += 0.10
        if any(k in t for k in ("срочно", "немедленно", "сейчас", "deadline", "дедлайн")):
            bonus += 0.15
        if any(k in t for k in ("спасибо", "благодар", "лучший", "обожаю", "восхищ")):
            bonus += 0.10
        if any(k in t for k in ("ужас", "отврат", "туп", "плохо", "зря")):
            bonus += 0.10
        return max(0.5, min(1.2, base + bonus))

    # ---------------- Labeling (Mahalanobis + penalties + hysteresis) ----------------
    def _score_mahal_label(self, pad: tuple[float, float, float]) -> tuple[str, float]:
        # если нет валидной карты — fallback на cosine
        if not self.use_multi or not isinstance(self.pad_multi, dict) or not self.pad_multi:
            lbl = self._cosine_label(pad, self.labeling.min_pad_norm)
            return lbl, 1.0  # фиктивный score

        best_lbl, best_s = "neutral", 1e9
        P, A, D = pad
        any_centroid = False

        for lbl, pack in self.pad_multi.items():
            if not isinstance(pack, dict):
                continue
            means = pack.get("means")
            covs = pack.get("covs")
            if not (isinstance(means, list) and isinstance(covs, list) and len(means) == len(covs) and len(means) > 0):
                continue

            smin = 1e9
            have_local = False
            for mu, diag in zip(means, covs):
                if not (isinstance(mu, list) and isinstance(diag, list) and len(mu) == 3 and len(diag) == 3):
                    continue
                try:
                    dP = (P - float(mu[0])) ** 2 / max(1e-6, float(diag[0]))
                    dA = (A - float(mu[1])) ** 2 / max(1e-6, float(diag[1]))
                    dD = (D - float(mu[2])) ** 2 / max(1e-6, float(diag[2]))
                except Exception:
                    continue
                smin = min(smin, dP + dA + dD)
                have_local = True
                any_centroid = True

            if not have_local:
                continue

            # квадрантные штрафы (эвристики)
            pen = 0.0
            if lbl == "sadness":
                if P > 0: pen += 0.6 * P
                if A > 0: pen += 0.4 * A
                if D > 0: pen += 0.4 * D
            elif lbl == "fear":
                if P > 0: pen += 0.6 * P
                if A < 0: pen += 0.4 * (-A)
                if D > 0: pen += 0.4 * D
            elif lbl == "anger":
                if P > 0: pen += 0.5 * P
                if A < 0: pen += 0.4 * (-A)
                if D < 0: pen += 0.3 * (-D)
            elif lbl == "joy":
                if P < 0: pen += 0.6 * (-P)
                if D < 0: pen += 0.4 * (-D)

            s = smin + 2.0 * pen
            if s < best_s:
                best_lbl, best_s = lbl, s

        if not any_centroid:
            # ничего валидного — безопасный откат
            lbl = self._cosine_label(pad, self.labeling.min_pad_norm)
            return lbl, 1.0

        return best_lbl, best_s

    def _score_mahal_for_label(self, pad: tuple[float, float, float], lbl: str) -> float:
        """
        Возвращает mahal-score (меньше — лучше) только для указанной метки.
        Если multi-прототипы недоступны, используем косинус: score = 1 - cos (меньше — лучше).
        """
        if not self.use_multi or lbl not in self.pad_multi:
            # cosine fallback
            Px, Ax, Dx = pad
            proto = PAD_PROTOS_DEFAULT.get(lbl)
            if not proto:
                return 1e9
            p, a, d = proto
            num = Px * p + Ax * a + Dx * d
            den = (math.sqrt(Px * Px + Ax * Ax + Dx * Dx) + 1e-12) * math.sqrt(p * p + a * a + d * d)
            if den <= 1e-12:
                return 1e9
            cosv = num / den
            return 1.0 - cosv  # меньше — лучше
        # multi-centroid
        P, A, D = pad
        pack = self.pad_multi.get(lbl)
        if not isinstance(pack, dict):
            return 1e9
        means = pack.get("means");
        covs = pack.get("covs")
        if not (isinstance(means, list) and isinstance(covs, list) and len(means) == len(covs) and len(means) > 0):
            return 1e9
        smin = 1e9
        for mu, diag in zip(means, covs):
            try:
                dP = (P - float(mu[0])) ** 2 / max(1e-6, float(diag[0]))
                dA = (A - float(mu[1])) ** 2 / max(1e-6, float(diag[1]))
                dD = (D - float(mu[2])) ** 2 / max(1e-6, float(diag[2]))
            except Exception:
                continue
            smin = min(smin, dP + dA + dD)
        if smin == 1e9:
            return 1e9
        # лёгкие квадрантные штрафы как в _score_mahal_label
        pen = 0.0
        if lbl == "sadness":
            if P > 0: pen += 0.6 * P
            if A > 0: pen += 0.4 * A
            if D > 0: pen += 0.4 * D
        elif lbl == "fear":
            if P > 0: pen += 0.6 * P
            if A < 0: pen += 0.4 * (-A)
            if D > 0: pen += 0.4 * D
        elif lbl == "anger":
            if P > 0: pen += 0.5 * P
            if A < 0: pen += 0.4 * (-A)
            if D < 0: pen += 0.3 * (-D)
        elif lbl == "joy":
            if P < 0: pen += 0.6 * (-P)
            if D < 0: pen += 0.4 * (-D)
        return smin + 2.0 * pen

    def _cosine_label(self, pad: Tuple[float,float,float], min_norm: float) -> str:
        if _norm3(pad) < min_norm:
            return "neutral"
        Px, Ax, Dx = pad
        best = ("neutral", -1.0)
        for lbl, (p, a, d) in PAD_PROTOS_DEFAULT.items():
            denom = (math.sqrt(p*p + a*a + d*d) * (_norm3(pad) + 1e-12))
            if denom <= 1e-12:
                continue
            cos = (Px*p + Ax*a + Dx*d) / denom
            if cos > best[1]:
                best = (lbl, cos)
        return best[0]

    def _label_with_hysteresis(
            self,
            pad: tuple[float, float, float],
            old_label: str,
            old_score: Optional[float]
    ) -> tuple[str, float]:
        """
        Возвращает (new_label, new_score).
        Гистерезис: принимаем новую метку, только если она заметно лучше старой.
        """
        # новый лучший по текущему паду
        new_label, new_score = self._score_mahal_label(pad) if (self.labeling.mode == "mahal" and self.use_multi) \
            else (self._cosine_label(pad, self.labeling.min_pad_norm), 0.0)

        # если раньше метки не было — просто принять новую
        if not old_label:
            return new_label, new_score

        # корректный score для СТАРОЙ метки относительно этого же pad
        try:
            old_score_exact = self._score_mahal_for_label(pad, old_label) if (
                        self.labeling.mode == "mahal" and self.use_multi) \
                else (1.0 - 1.0)  # в cosine-ветке score не используется
        except Exception:
            old_score_exact = 1e9

        # порог гистерезиса: новая должна быть лучше старой на hysteresis * scale
        h = self.labeling.hysteresis
        improve = (old_score_exact - new_score)
        threshold = h * max(1.0, old_score_exact)

        if improve > threshold:
            return new_label, new_score
        else:
            # оставляем старую, возвращаем её «точный» скор
            return old_label, old_score_exact

    # ---------------- β ----------------
    def _conf(self, p_top: float, Hn: float, t: float = 0.50, k: float = 10.0) -> float:
        z = 0.5 * p_top + 0.5 * (1.0 - Hn)
        return _sigmoid(k * (z - t))

    def _beta_active(self, coeff_conf: float) -> float:
        """β при событии: база + уверенность + свежесть + импульс, минимум hold_beta в hold-окно."""
        beta = (self.time_cfg.beta_base
                + self.time_cfg.beta_entropy_gain * coeff_conf
                + self.time_cfg.beta_burst_gain * self.state.burst_mark
                + self.time_cfg.beta_impulse_gain * self.state.impulse_mark)
        if self.state.hold_left_s > 0:
            beta = max(beta, self.time_cfg.hold_beta)
        beta = max(0.0, min(1.0, beta))
        self.state.beta_current = beta
        return beta

    def _beta_idle(self) -> float:
        """β в простое, плавное сходение из hold к idle на пост-холд τ."""
        if self.state.hold_left_s > 0:
            beta = self.time_cfg.hold_beta
        else:
            tail = math.exp(- self.state.post_hold_elapsed_s / max(1e-6, self.time_cfg.post_hold_tau_s))
            base_idle = self.time_cfg.beta_base_idle \
                        + self.time_cfg.beta_burst_gain * self.state.burst_mark \
                        + 0.5 * self.time_cfg.beta_impulse_gain * self.state.impulse_mark
            beta = base_idle + tail * (self.time_cfg.hold_beta - base_idle)
        beta = max(0.0, min(1.0, beta))
        self.state.beta_current = beta
        return beta

    def _recompute_eff(self, beta: float) -> None:
        mP, mA, mD = self.state.mood
        bP, bA, bD = self.state.burst
        eff = ((1.0 - beta) * mP + beta * bP,
               (1.0 - beta) * mA + beta * bA,
               (1.0 - beta) * mD + beta * bD)
        lim = self.layers.pad_bounds
        self.state.eff = (max(-lim, min(lim, eff[0])),
                          max(-lim, min(lim, eff[1])),
                          max(-lim, min(lim, eff[2])))

    # ---------------- dynamics ----------------
    def idle_tick(self, dt_s: float) -> None:
        if dt_s <= 0:
            return

        # hold window и post-hold счётчик
        if self.state.hold_left_s > 0:
            self.state.hold_left_s = max(0.0, self.state.hold_left_s - dt_s)
            self.state.post_hold_elapsed_s = 0.0
            if self.state.hold_left_s == 0.0:
                self.state.last_hold_end_ts = time.time()
        else:
            self.state.post_hold_elapsed_s += dt_s

        # burst распад
        k_b = _exp_decay_factor_half_life(dt_s, self.time_cfg.burst_half_life_s)
        bP, bA, bD = self.state.burst
        self.state.burst = (bP * k_b, bA * k_b, bD * k_b)
        # marks: EMA подъём/спад
        normb = _norm3(self.state.burst)
        self.state.burst_mark = _ema(self.state.burst_mark, normb, 0.30)  # быстрый подъём/медленн. спад через EMA к 0
        self.state.burst_mark *= _exp_decay_factor_half_life(dt_s, self.time_cfg.beta_burst_half_life_s)
        self.state.impulse_mark *= _exp_decay_factor_half_life(dt_s, self.time_cfg.beta_impulse_half_life_s)

        # fatigue
        thr = float(self.fatigue_cfg.get("threshold_A", 0.40))
        gain = float(self.fatigue_cfg.get("gain_per_sec", 0.015))
        relax_hl = float(self.fatigue_cfg.get("relax_half_life_s", 900.0))
        if self.state.mood[1] > thr:
            self.state.fatigue = min(1.0, self.state.fatigue + gain * dt_s)
        else:
            self.state.fatigue *= _exp_decay_factor_half_life(dt_s, relax_hl)

        # mood → baseline
        k_m = _exp_decay_factor_half_life(dt_s, self.time_cfg.mood_decay_half_life_s)
        base = (self.persona.baseline_P, self.persona.baseline_A, self.persona.baseline_D)
        mP, mA, mD = self.state.mood
        self.state.mood = (
            _lerp(base[0], mP, k_m),
            _lerp(base[1], mA, k_m),
            _lerp(base[2], mD, k_m)
        )

        # mood evidence decay (медленный распад интегратора)
        k_ev = _exp_decay_factor_half_life(dt_s, max(1e-6, self.mood_ev_hl))
        evP, evA, evD = self.state.mood_ev
        self.state.mood_ev = (evP * k_ev, evA * k_ev, evD * k_ev)

        # fatigue impact (легкий сдвиг eff через mood)
        imp = self.fatigue_cfg.get("impact", {"A":-0.30,"P":-0.05,"D":-0.10})
        f = self.state.fatigue
        if f > 0.0:
            self.state.mood = (
                self.state.mood[0] + imp.get("P",-0.05)*f*0.05,
                self.state.mood[1] + imp.get("A",-0.30)*f*0.05,
                self.state.mood[2] + imp.get("D",-0.10)*f*0.05
            )

        # eff к mood c β_idle
        beta_idle = self._beta_idle()
        self._recompute_eff(beta_idle)

        # метки
        self._update_labels()

        # time
        now = time.time()
        self.state.last_ts = self.state.ts
        self.state.ts = now

    def step(self, text: str) -> KriplAffectState:
        now = time.time()
        dt = max(0.0, now - self.state.ts)
        if dt > 0:
            self.idle_tick(dt)

        # domain / context / len
        domain = self._detect_domain(text)
        ctx = self._context_factor(text)
        ln = _len_norm(text)
        self.state.domain = domain
        self.state.context_factor = ctx
        self.state.len_norm = ln

        # 1) backend perception
        out = self.backend.predict(text)
        probs = out.probs
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_lbl, top_prob = ranked[0]
        Hn = _entropy_norm([probs[lbl] for lbl in self.labels])

        # 2) pad_user anchor
        if self.backend_cfg.use_top_only:
            p,a,d = PAD_PROTOS_DEFAULT.get(top_lbl, (0.0, 0.0, 0.0))
            pad_user = (p * float(top_prob), a * float(top_prob), d * float(top_prob))
        else:
            pad_user = self._pad_from_mix(probs)

        # soft neutralization by confidence (no hard zero)
        conf = self._conf(float(top_prob), float(Hn), t=0.50, k=10.0)
        scale = 0.2 + 0.8 * conf
        pad_user = tuple(v * scale for v in pad_user)

        # length & context scaling
        pad_user = tuple(v * (0.6 + 0.4 * ln) * ctx for v in pad_user)

        # domain bias (малые поправки)
        bias = self.domain_bias.get(domain, {"P":0.0,"A":0.0,"D":0.0})
        pad_user = (pad_user[0] + bias.get("P",0.0)*0.08,
                    pad_user[1] + bias.get("A",0.0)*0.08,
                    pad_user[2] + bias.get("D",0.0)*0.08)

        # 3) ΔPAD_K (эмпатия)
        dP, dA, dD = self._empathy_transform(pad_user)

        # 4) burst — быстрый рост к ΔPAD_K
        rise = _exp_rise_factor_tau(dt_s=max(1e-3, dt), tau_s=self.time_cfg.burst_rise_tau_s)
        bP, bA, bD = self.state.burst
        bP = bP + rise * (dP - bP)
        bA = bA + rise * (dA - bA)
        bD = bD + rise * (dD - bD)
        self.state.burst = (_clip(bP, self.layers.burst_clip),
                            _clip(bA, self.layers.burst_clip),
                            _clip(bD, self.layers.burst_clip))

        # marks (EMA + импульс)
        normb = _norm3(self.state.burst)
        self.state.burst_mark = _ema(self.state.burst_mark, normb, 0.40)
        self.state.impulse_mark = 1.0

        # hold window + refractory
        hold = self.time_cfg.hold_seconds * (0.6 + 0.4 * ln) * ctx
        since_last_hold_end = now - self.state.last_hold_end_ts
        if since_last_hold_end < self.time_cfg.refractory_s:
            hold *= self.time_cfg.refractory_hold_scale
            self.time_cfg.hold_beta * self.time_cfg.refractory_beta_scale  # эффект на β применится через _beta_active
        self.state.hold_left_s = max(self.state.hold_left_s, hold)

        # --- mood evidence integrator update ---
        # как сильно собираем «доказательства»: учитываем длину/контекст
        k_acc = self.mood_ev_gain * (0.6 + 0.4 * ln) * (1.0 + self.mood_ev_ctx * (ctx - 1.0))
        # аккуратное ограничение шага (на случай больших dt)
        k_acc = max(0.0, min(1.0, k_acc))
        evP, evA, evD = self.state.mood_ev
        evP = evP + k_acc * (pad_user[0] - evP)
        evA = evA + k_acc * (pad_user[1] - evA)
        evD = evD + k_acc * (pad_user[2] - evD)
        # клип интегратора
        c = self.mood_ev_clip
        self.state.mood_ev = (_clip(evP, c), _clip(evA, c), _clip(evD, c))


        # 5) mood — медленное следование к baseline + слабое следование пользователю
        base = (self.persona.baseline_P, self.persona.baseline_A, self.persona.baseline_D)
        target_mood = (
            base[0] + self.empathy.mood_follow_user * self.mood_gain * pad_user[0] + self.state.mood_ev[0],
            base[1] + self.empathy.mood_follow_user * self.mood_gain * pad_user[1] + self.state.mood_ev[1],
            base[2] + self.empathy.mood_follow_user * self.mood_gain * pad_user[2] + self.state.mood_ev[2]
        )

        mP, mA, mD = self.state.mood
        m_rise = _exp_rise_factor_tau(dt_s=max(1e-3, dt), tau_s=self.time_cfg.mood_rise_tau_s)
        mP = mP + m_rise * (target_mood[0] - mP)
        mA = mA + m_rise * (target_mood[1] - mA)
        mD = mD + m_rise * (target_mood[2] - mD)
        self.state.mood = (_clip(mP, self.layers.mood_clip),
                           _clip(mA, self.layers.mood_clip),
                           _clip(mD, self.layers.mood_clip))

        # 6) eff — активный β с приоритетом burst (hold)
        beta = self._beta_active(conf)
        self._recompute_eff(beta)

        # micro-noise для живости (малый)
        if self.micro_noise_std > 0:
            rn = lambda: random.gauss(0.0, self.micro_noise_std)
            self.state.eff = (_clip(self.state.eff[0] + rn(), 1.0),
                              _clip(self.state.eff[1] + rn(), 1.0),
                              _clip(self.state.eff[2] + rn(), 1.0))

        # labels (через калибровку и hysteresis)
        self._update_labels()

        # save USER meta
        self.state.user_text = text
        self.state.user_top = top_lbl
        self.state.user_top_prob = float(top_prob)
        self.state.user_entropy = float(Hn)
        self.state.user_pad = pad_user

        self.state.last_ts = self.state.ts
        self.state.ts = now
        return self.state

    # ---------------- helpers ----------------
    def _empathy_transform(self, pad_u: Tuple[float, float, float]) -> Tuple[float, float, float]:
        Pu, Au, Du = pad_u
        E = self.empathy.E
        W = self.empathy.W_empathy * getattr(self, "burst_gain", 1.0)  # усиление чувствительности burst
        dP = W * (E.get("P_from_P", 0.0) * Pu + E.get("P_from_A", 0.0) * Au + E.get("P_from_D", 0.0) * Du)
        dA = W * (E.get("A_from_P", 0.0) * Pu + E.get("A_from_A", 0.0) * Au + E.get("A_from_D", 0.0) * Du)
        dD = W * (E.get("D_from_P", 0.0) * Pu + E.get("D_from_A", 0.0) * Au + E.get("D_from_D", 0.0) * Du)
        dP = _clip(dP, self.limits.pad_clip_per_step)
        dA = _clip(dA, self.limits.pad_clip_per_step)
        dD = _clip(dD, self.limits.pad_clip_per_step)
        return (dP, dA, dD)

    def _final_pad_for_label(self, pad_eff: Tuple[float,float,float],
                             pad_anchor: Tuple[float,float,float]) -> Tuple[float,float,float]:
        # якорение: смешиваем eff с якорем от восприятия перед лейблом (уменьшает «улёты»)
        alpha = 0.65
        P = alpha * pad_eff[0] + (1-alpha) * pad_anchor[0]
        A = alpha * pad_eff[1] + (1-alpha) * pad_anchor[1]
        D = alpha * pad_eff[2] + (1-alpha) * pad_anchor[2]
        return self._pad_calibrate((P,A,D))

    def _update_labels(self) -> None:
        # собрать final_pad для каждого слоя относительно текущего пользовательского якоря
        anchor = self.state.user_pad  # уже масштабирован и сдвинут
        # mood
        pad_m = self._final_pad_for_label(self.state.mood, anchor)
        # burst
        pad_b = self._final_pad_for_label(self.state.burst, anchor)
        # eff
        pad_e = self._final_pad_for_label(self.state.eff, anchor)

        # norms -> neutral guard
        if _norm3(pad_m) < self.labeling.min_pad_norm: self.state.mood_label = "neutral"
        else:
            self.state.mood_label, _ = self._label_with_hysteresis(pad_m, self.state.mood_label, None)
        if _norm3(pad_b) < self.labeling.min_pad_norm: self.state.burst_label = "neutral"
        else:
            self.state.burst_label, _ = self._label_with_hysteresis(pad_b, self.state.burst_label, None)
        if _norm3(pad_e) < self.labeling.min_pad_norm: self.state.eff_label = "neutral"
        else:
            self.state.eff_label, _ = self._label_with_hysteresis(pad_e, self.state.eff_label, None)

    # ---------------- snapshot ----------------
    def snapshot(self) -> Dict[str, Any]:
        s = self.state
        return {
            "USER": {
                "text": s.user_text,
                "top_label": s.user_top,
                "top_prob": s.user_top_prob,
                "entropy_norm": s.user_entropy,
                "pad_user": {"P": s.user_pad[0], "A": s.user_pad[1], "D": s.user_pad[2]}
            },
            "KRIPL": {
                "mood": {"P": s.mood[0], "A": s.mood[1], "D": s.mood[2], "label": s.mood_label},
                "burst": {"P": s.burst[0], "A": s.burst[1], "D": s.burst[2], "label": s.burst_label},
                "eff": {"P": s.eff[0], "A": s.eff[1], "D": s.eff[2], "label": s.eff_label},
                "marks": {
                    "burst": s.burst_mark,
                    "impulse": s.impulse_mark,
                    "hold_left_s": s.hold_left_s,
                    "post_hold_elapsed_s": s.post_hold_elapsed_s,
                    "beta_current": s.beta_current,
                    "fatigue": s.fatigue,
                    "domain": s.domain,
                    "context_factor": s.context_factor,
                    "len_norm": s.len_norm
                },
                "ts": s.ts, "last_ts": s.last_ts
            }
        }
