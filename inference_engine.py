"""
Tier 4: Model Serving Layer (inference_engine.py)
Implements the soft-voting ensemble of:
  1. XGBoost operating on the 393-dimensional multi-tier feature vector
  2. DistilBERT classifier served via ONNX Runtime with ORT_ENABLE_ALL graph optimization
     (with automatic fallback to a trained neural sequence classifier if ONNX model is not yet exported)
"""

import json
import os
import pickle
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class InferenceEngine:
    def __init__(
        self,
        onnx_path="models/distilbert.onnx",
        xgb_path="models/xgb.json",
        fallback_bert_path="models/bert_fallback.pkl",
        w=0.5,
        t_lo=0.35,
        t_hi=0.85,
    ):
        self.w = w
        self.t_lo = t_lo
        self.t_hi = t_hi
        if os.path.exists("models/calibration.json"):
            try:
                with open("models/calibration.json", "r", encoding="utf-8") as cf:
                    cal = json.load(cf)
                    self.w = float(cal.get("w", self.w))
                    self.t_lo = float(cal.get("t_lo", self.t_lo))
                    self.t_hi = float(cal.get("t_hi", self.t_hi))
            except Exception:
                pass
        self.sess = None
        self.tok = None
        self.booster = None
        self.fallback_bert = None
        self.fallback_xgb = None

        # 1. Load ONNX Runtime session if distilbert.onnx exists
        if ort is not None and os.path.exists(onnx_path):
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            opts.intra_op_num_threads = 4
            available = ort.get_available_providers()
            providers = [
                p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if p in available
            ] or ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(onnx_path, opts, providers=providers)
            if AutoTokenizer is not None:
                try:
                    self.tok = AutoTokenizer.from_pretrained(
                        "distilbert-base-uncased"
                    )
                except Exception:
                    self.tok = None

        # Load calibrated neural sequence classifier if ONNX file isn't exported yet
        if self.sess is None and os.path.exists(fallback_bert_path):
            with open(fallback_bert_path, "rb") as f:
                self.fallback_bert = pickle.load(f)

        # 2. Load XGBoost booster
        if xgb is not None and os.path.exists(xgb_path):
            self.booster = xgb.Booster()
            self.booster.load_model(xgb_path)
        elif os.path.exists("models/xgb_fallback.pkl"):
            with open("models/xgb_fallback.pkl", "rb") as f:
                self.fallback_xgb = pickle.load(f)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def p_bert(self, text: str) -> float:
        """P(attack) from the ONNX DistilBERT classifier (or neural sequence model)."""
        if self.sess is not None and self.tok is not None:
            enc = self.tok(
                text,
                return_tensors="np",
                truncation=True,
                max_length=256,
                padding="max_length",
            )
            feed = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            logits = self.sess.run(None, feed)[0]
            return float(self._softmax(logits)[0, 1])

        if self.fallback_bert is not None:
            prob = self.fallback_bert.predict_proba([text])[0, 1]
            return float(prob)

        # Heuristic contextual fallback if no model is trained yet
        return 0.5

    def p_xgb(self, feats: np.ndarray) -> float:
        """P(attack) from XGBoost on the 393-d engineered feature vector."""
        if self.booster is not None and xgb is not None:
            d = xgb.DMatrix(feats.reshape(1, -1))
            return float(self.booster.predict(d)[0])

        if self.fallback_xgb is not None:
            prob = self.fallback_xgb.predict_proba(feats.reshape(1, -1))[0, 1]
            return float(prob)

        # Heuristic feature fallback if no model is trained yet
        trig = feats[2]
        delim = feats[5]
        homo = feats[7]
        raw_score = min(1.0, 0.35 * trig + 0.20 * delim + 0.15 * homo)
        return float(raw_score)

    def predict(self, text: str, feats: np.ndarray) -> dict:
        """Soft voting: p = w * p_x + (1 - w) * p_b."""
        px = self.p_xgb(feats)
        pb = self.p_bert(text)
        p = self.w * px + (1.0 - self.w) * pb

        if p >= self.t_hi:
            action = "BLOCK"
        elif p >= self.t_lo:
            action = "SANITIZE"
        else:
            action = "PASS"

        return {
            "p": round(float(p), 4),
            "p_xgb": round(float(px), 4),
            "p_bert": round(float(pb), 4),
            "action": action,
        }
