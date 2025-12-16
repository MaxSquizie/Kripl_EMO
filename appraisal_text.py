# -*- coding: utf-8 -*-
"""
Unified text appraisal (OFFLINE) for Cripl EMO.

Backends:
  - 'goemo_ru' (default offline): local ru GoEmotions model directory only.

Usage:
  app = TextAppraisal(backend_opts={'local_dir': r'C:\\AI\\emo_model', 'device': 'cuda'})
  res = app.analyze("Спасибо!", state={'P':0.1,'A':0.0,'D':0.0})
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os, traceback

# Force offline
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from emo_backend_goemotions import GoEmotionsRU, dpad_from_target


class TextAppraisal:
    """
    analyze(text, state={'P','A','D'}) -> { 'deltas':{dP,dA,dD}, 'metrics':{...} }

    backend_opts:
      - local_dir: REQUIRED path to local model dir (e.g. r"C:\\AI\\emo_model") if env not set
      - device:    'cuda' | 'cpu'  (optional; default picks automatically)
    """
    def __init__(self,
                 backend: Optional[str] = None,
                 backend_opts: Optional[Dict[str, Any]] = None) -> None:
        self.backend = backend or os.environ.get("CRIPL_EMO_BACKEND", "goemo_ru")
        self.backend_opts = backend_opts or {}
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "goemo_ru":
            self.model = GoEmotionsRU(
                local_dir=self.backend_opts.get("local_dir") or os.environ.get("CRIPL_EMO_LOCALDIR"),
                device=self.backend_opts.get("device") or os.environ.get("CRIPL_EMO_DEVICE")
            )
            return
        raise ValueError(f"Unknown EMO backend: {self.backend}")

    def analyze(self, text: str, state: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        state = state or {"P": 0.0, "A": 0.0, "D": 0.0}
        try:
            out = self.model.predict(text)
            target = out.pad_target
            cur = (float(state["P"]), float(state["A"]), float(state["D"]))
            dpad = dpad_from_target(target, cur)

            metrics = {
                "mode": out.mode,
                "probs": out.probs,
                "active": out.active,
                "pad_target": {"P": target[0], "A": target[1], "D": target[2]},
                "intensity_raw": out.intensity_raw,
                "intensity_cal": out.intensity_cal,
                "sentiment_score_raw": out.sentiment_score_raw,
                "sentiment_score_cal": out.sentiment_score_cal,
            }
            return {"deltas": dpad, "metrics": metrics}
        except Exception as e:
            tb = traceback.format_exc()
            metrics = {"mode": self.backend, "error": str(e), "traceback": tb}
            # Fail-safe zero Δ
            return {"deltas": {"dP": 0.0, "dA": 0.0, "dD": 0.0}, "metrics": metrics}
