# -*- coding: utf-8 -*-
"""
GoEmotions (ru) backend for Cripl EMO (STRICTLY OFFLINE):
- Uses ONLY a local model directory (no internet).
- Model expected: ruRoberta-large-ru-go-emotions (or compatible) placed locally.

Required files inside local_dir (typical HuggingFace layout):
  - config.json (with correct id2label/label2id)
  - tokenizer.json (+ tokenizer_config.json, vocab.json/merges.txt if BPE)
  - model.safetensors (or pytorch_model.bin)
Optional:
  - best_thresholds.pkl  (per-class thresholds dict[str->float])

Set env before running (or pass in backend_opts):
  CRIPL_EMO_LOCALDIR = C:\\AI\\emo_model
  CRIPL_EMO_DEVICE   = cuda | cpu
  TRANSFORMERS_OFFLINE=1
  HF_HUB_OFFLINE=1
"""

from __future__ import annotations
import os, math, pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Force offline mode for transformers/hf hub
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Device & runtime knobs
DEFAULT_DEVICE = os.environ.get("CRIPL_EMO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMO_TEMP = float(os.environ.get("CRIPL_EMO_TEMP", "1.15"))

# PAD gain & clipping
GAIN_PAD = {
    "P": float(os.environ.get("CRIPL_GAIN_P", "0.50")),
    "A": float(os.environ.get("CRIPL_GAIN_A", "0.65")),
    "D": float(os.environ.get("CRIPL_GAIN_D", "0.50")),
}
DELTA_CLIP = float(os.environ.get("CRIPL_DELTA_CLIP", "0.75"))

# Default PAD prototypes for GoEmotions labels (approximate; range [-1,1])
PAD_PROTOS_DEFAULT: Dict[str, Tuple[float, float, float]] = {
    "admiration":     (+0.60, +0.35, +0.30),
    "amusement":      (+0.65, +0.55, +0.10),
    "anger":          (-0.70, +0.60, +0.35),
    "annoyance":      (-0.40, +0.40, +0.10),
    "approval":       (+0.45, +0.20, +0.15),
    "caring":         (+0.55, +0.10, +0.10),
    "confusion":      (-0.10, +0.35, -0.15),
    "curiosity":      (+0.25, +0.55, -0.05),
    "desire":         (+0.50, +0.60, +0.15),
    "disappointment": (-0.55, -0.15, -0.20),
    "disapproval":    (-0.55, +0.25, +0.10),
    "disgust":        (-0.75, +0.45, +0.25),
    "embarrassment":  (-0.55, +0.40, -0.40),
    "excitement":     (+0.70, +0.85, +0.30),
    "fear":           (-0.80, +0.80, -0.60),
    "gratitude":      (+0.70, +0.30, +0.10),
    "grief":          (-0.85, -0.40, -0.50),
    "joy":            (+0.85, +0.60, +0.40),
    "love":           (+0.80, +0.45, +0.35),
    "nervousness":    (-0.35, +0.70, -0.35),
    "optimism":       (+0.60, +0.35, +0.20),
    "pride":          (+0.65, +0.40, +0.70),
    "realization":    (+0.10, +0.25, +0.10),
    "relief":         (+0.55, -0.20, +0.30),
    "remorse":        (-0.70, +0.25, -0.45),
    "sadness":        (-0.75, -0.35, -0.35),
    "surprise":       (+0.10, +0.90, +0.05),
    "neutral":        ( 0.00,  0.00,  0.00),
}

POS = {"admiration","amusement","approval","caring","curiosity","desire",
       "excitement","gratitude","joy","love","optimism","pride","relief"}
NEG = {"anger","annoyance","disappointment","disapproval","disgust","embarrassment",
       "fear","grief","nervousness","remorse","sadness"}


@dataclass
class GoEmoOutput:
    probs: Dict[str, float]
    active: List[str]
    pad_target: Tuple[float, float, float]
    dpad: Dict[str, float]
    intensity_raw: float
    intensity_cal: float
    sentiment_score_raw: float
    sentiment_score_cal: float
    mode: str


class GoEmotionsRU:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None, local_dir: Optional[str] = None):
        """
        OFFLINE ONLY. One of (local_dir or CRIPL_EMO_LOCALDIR env) MUST be provided.
        """
        self.device = device or DEFAULT_DEVICE
        self.local_dir = local_dir or os.environ.get("CRIPL_EMO_LOCALDIR")
        if not self.local_dir:
            raise RuntimeError(
                "GoEmotionsRU requires a local model directory. "
                "Set backend_opts={'local_dir': r'C:\\AI\\emo_model'} "
                "or env CRIPL_EMO_LOCALDIR to the local path. (No internet fallback.)"
            )
        if not os.path.isdir(self.local_dir):
            raise FileNotFoundError(f"Local model directory not found: {self.local_dir}")

        # Strict local loading
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.local_dir, local_files_only=True
        )
        self.model.to(self.device).eval()

        # Dynamic label order from config.id2label (strictly local)
        self.order = self._labels_from_config()
        self.pad_protos = self._align_pad_protos(self.order)
        self.thresholds = self._load_thresholds(self.local_dir, self.order)

    # ----- config helpers -----
    def _labels_from_config(self) -> List[str]:
        cfg = getattr(self.model, "config", None)
        if not cfg or not hasattr(cfg, "id2label") or not isinstance(cfg.id2label, dict) or len(cfg.id2label) == 0:
            # Fall back to default GoEmotions order if no id2label
            return [
                "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
                "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
                "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
                "remorse","sadness","surprise","neutral"
            ]
        # id2label keys can be "0","1",... or ints
        pairs = []
        for k, v in cfg.id2label.items():
            try:
                pairs.append((int(k), v))
            except Exception:
                pairs.append((k, v))
        pairs = sorted(pairs, key=lambda x: x[0])
        return [lbl for _, lbl in pairs]

    def _align_pad_protos(self, labels: List[str]) -> Dict[str, Tuple[float, float, float]]:
        return {lbl: PAD_PROTOS_DEFAULT.get(lbl, PAD_PROTOS_DEFAULT["neutral"]) for lbl in labels}

    def _load_thresholds(self, src_path: str, labels: List[str]) -> Dict[str, float]:
        # 1) src_path/best_thresholds.pkl
        cand = os.path.join(src_path, "best_thresholds.pkl")
        if os.path.isfile(cand):
            try:
                with open(cand, "rb") as f:
                    th = pickle.load(f)
                return {lbl: float(th.get(lbl, 0.50)) for lbl in labels}
            except Exception:
                pass
        # 2) src_path/onnx/best_thresholds.pkl
        cand2 = os.path.join(src_path, "onnx", "best_thresholds.pkl")
        if os.path.isfile(cand2):
            try:
                with open(cand2, "rb") as f:
                    th = pickle.load(f)
                return {lbl: float(th.get(lbl, 0.50)) for lbl in labels}
            except Exception:
                pass
        # Defaults if thresholds file absent
        base = {k: 0.50 for k in labels}
        for k in ("joy","love","gratitude","admiration","amusement","excitement","optimism","pride","relief"):
            if k in base: base[k] = 0.40
        for k in ("grief","remorse","disgust","disapproval"):
            if k in base: base[k] = 0.55
        return base

    # ----- math helpers -----
    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    @staticmethod
    def _entropy(p: List[float]) -> float:
        eps = 1e-12
        return -sum(pi * math.log(max(pi, eps)) for pi in p)

    def _temp_scale(self, probs: List[float], temp: float) -> List[float]:
        if temp <= 1.001:
            return probs
        q = [p ** (1.0 / temp) for p in probs]
        s = sum(q) + 1e-12
        return [v / s for v in q]

    # ----- public -----
    @torch.inference_mode()
    def predict(self, text: str) -> GoEmoOutput:
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        logits: torch.Tensor = self.model(**toks).logits.squeeze(0)  # (n_labels,)

        probs = self._sigmoid(logits).detach().cpu().tolist()
        probs_map = {lbl: float(probs[i]) for i, lbl in enumerate(self.order)}

        # active labels via thresholds
        active = [lbl for lbl in self.order if probs_map[lbl] >= self.thresholds.get(lbl, 0.5)]

        # temperature mixing for PAD target (allows mixed emotions)
        probs_t = self._temp_scale([probs_map[k] for k in self.order], temp=EMO_TEMP)
        probs_t_map = {k: probs_t[i] for i, k in enumerate(self.order)}

        # target PAD (neutral contributes less)
        w_sum = 1e-9
        tgtP = tgtA = tgtD = 0.0
        for lbl, w in probs_t_map.items():
            w_eff = w * (0.25 if lbl == "neutral" else 1.0)
            p, a, d = self.pad_protos.get(lbl, PAD_PROTOS_DEFAULT["neutral"])
            tgtP += w_eff * p; tgtA += w_eff * a; tgtD += w_eff * d
            w_sum += w_eff
        tgtP /= w_sum; tgtA /= w_sum; tgtD /= w_sum

        # proxy valence & intensity
        pos_sum = sum(probs_map.get(k, 0.0) for k in POS)
        neg_sum = sum(probs_map.get(k, 0.0) for k in NEG)
        sent_raw = (pos_sum - neg_sum) / max(pos_sum + neg_sum + 1e-9, 1.0)

        maxp = max(probs_map.values()) if probs_map else 0.0
        H = self._entropy(list(probs_map.values()))
        H_max = math.log(max(len(self.order), 1))
        inten = 0.6 * maxp + 0.4 * (1.0 - H / H_max)

        sent_cal = max(-1.0, min(1.0, 1.15 * sent_raw))
        inten_cal = max(0.0, min(1.0, inten))

        return GoEmoOutput(
            probs=probs_map,
            active=active,
            pad_target=(tgtP, tgtA, tgtD),
            dpad={"dP": 0.0, "dA": 0.0, "dD": 0.0},
            intensity_raw=float(maxp),
            intensity_cal=float(inten_cal),
            sentiment_score_raw=float(sent_raw),
            sentiment_score_cal=float(sent_cal),
            mode="goemotions",
        )


def dpad_from_target(target_pad: Tuple[float, float, float],
                     current_pad: Tuple[float, float, float]) -> Dict[str, float]:
    """Convert target PAD into a ΔPAD step towards it, with per-axis gains and clipping."""
    tP, tA, tD = target_pad
    cP, cA, cD = current_pad
    dP = (tP - cP) * GAIN_PAD["P"]
    dA = (tA - cA) * GAIN_PAD["A"]
    dD = (tD - cD) * GAIN_PAD["D"]

    def _clip(x: float) -> float:
        return max(-DELTA_CLIP, min(DELTA_CLIP, x))

    return {"dP": _clip(dP), "dA": _clip(dA), "dD": _clip(dD)}
