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
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class InferenceEngine:
    def __init__(
        self,
        onnx_path="models/distilbert.onnx",
        xgb_path="models/xgb.json",
        fallback_bert_path="models/bert_fallback.pkl",
        w_bert=0.75,
        w_xgb=0.25,
        t_lo=0.35,
        t_hi=0.85,
    ):
        self.w_bert = w_bert
        self.w_xgb = w_xgb
        self.w = w_xgb
        self.t_lo = t_lo
        self.t_hi = t_hi
        if os.path.exists("models/calibration.json"):
            try:
                with open("models/calibration.json", "r", encoding="utf-8") as cf:
                    cal = json.load(cf)
                    self.w_bert = float(cal.get("w_bert", 0.75))
                    self.w_xgb = float(cal.get("w_xgb", cal.get("w", 0.25)))
                    self.w = self.w_xgb
                    self.t_lo = float(cal.get("t_lo", self.t_lo))
                    self.t_hi = float(cal.get("t_hi", self.t_hi))
            except Exception:
                pass
        self.sess = None
        self.tok = None
        self.booster = None
        self.fallback_bert = None
        self.fallback_xgb = None
        self.hf_model = None
        self.hf_tok = None

        # 1. Primary Transformer: Load local Hugging Face DeBERTa/BERT model if available
        if AutoTokenizer is not None and AutoModelForSequenceClassification is not None and torch is not None:
            try:
                hf_name = "protectai/deberta-v3-base-prompt-injection-v2"
                self.hf_tok = AutoTokenizer.from_pretrained(hf_name)
                self.hf_model = AutoModelForSequenceClassification.from_pretrained(hf_name)
                self.hf_model.eval()
            except Exception:
                self.hf_model = None
                self.hf_tok = None

        # 2. Load ONNX Runtime session if distilbert.onnx exists
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

        # 3. Load calibrated neural sequence classifier (MLP on n-grams) as lightweight fallback
        if os.path.exists(fallback_bert_path):
            try:
                with open(fallback_bert_path, "rb") as f:
                    self.fallback_bert = pickle.load(f)
            except Exception:
                self.fallback_bert = None

        # 4. Load XGBoost booster (Secondary Structural Feature Verifier)
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
        """P(attack) from the Primary BERT/DeBERTa classifier (with ONNX & MLP sequence fallbacks)."""
        # Primary: PyTorch DeBERTa/BERT prompt injection classifier
        if self.hf_model is not None and self.hf_tok is not None and torch is not None:
            try:
                inputs = self.hf_tok(text, return_tensors="pt", truncation=True, max_length=256)
                with torch.no_grad():
                    logits = self.hf_model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                    return float(probs[1])
            except Exception:
                pass

        # ONNX runtime accelerated sequence inference
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

        # Neural sequence model fallback
        if self.fallback_bert is not None:
            prob = self.fallback_bert.predict_proba([text])[0, 1]
            return float(prob)

        return 0.5

    def p_xgb(self, feats) -> float:
        """P(attack) from XGBoost on the 393-d engineered feature vector (Secondary Verifier)."""
        feats_arr = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        if self.booster is not None and xgb is not None:
            d = xgb.DMatrix(feats_arr)
            return float(self.booster.predict(d)[0])

        if self.fallback_xgb is not None:
            prob = self.fallback_xgb.predict_proba(feats_arr)[0, 1]
            return float(prob)

        # Heuristic feature fallback if no model is trained yet
        trig = feats[2]
        delim = feats[5]
        homo = feats[7]
        raw_score = min(1.0, 0.35 * trig + 0.20 * delim + 0.15 * homo)
        return float(raw_score)

    def predict(self, text: str, feats: np.ndarray) -> dict:
        """
        Primary BERT Classifier with Secondary XGBoost Structural Verifier:
        p = w_bert * p_b + w_xgb * p_x
        Default weights: BERT = 75%, XGBoost = 25%
        """
        px = self.p_xgb(feats)
        pb = self.p_bert(text)

        # Soft voting weighted towards primary BERT model
        p = self.w_bert * pb + self.w_xgb * px

        # If BERT has high confidence (>= 0.82), ensure it drives the block decision
        if pb >= 0.82:
            p = max(p, pb)

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
            "primary_model": "BERT/DeBERTa",
            "w_bert": self.w_bert,
            "w_xgb": self.w_xgb,
        }
