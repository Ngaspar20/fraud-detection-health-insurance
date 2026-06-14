"""
supervised_model.py
-------------------
Stage-2 supervised classifier that sits on top of the Isolation Forest.

Pipeline:
  Stage 1 (fraud_detection.py) → flags anomalies (unsupervised, no labels needed)
  Stage 2 (this module)        → predicts fraud probability using investigator
                                  feedback as training labels (supervised)

The classifier is a Random Forest trained on the composite sub-scores plus
raw claim features. It requires at least MIN_SAMPLES confirmed verdicts
(Fraude Confirmada / Falso Positivo) before it will train.

Model is persisted to data/fraud_classifier.pkl so it survives app restarts
within the same session. On Streamlit Cloud, the file is ephemeral across
redeploys — the user must retrain after each deployment using the Retrain button.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

MODEL_PATH = Path(__file__).parent.parent / "data" / "fraud_classifier.pkl"

# Minimum labeled examples (of each class) before training is allowed
MIN_SAMPLES = 5   # low threshold so demo works with limited feedback

# Features fed into the supervised model
FEATURE_COLS = [
    "anomaly_score",
    "provider_risk_score",
    "member_risk_score",
    "cost_outlier_score",
    "risk_score",
    "claim_amount",
    "flag_count",
]

FEATURE_LABELS = {
    "anomaly_score":       "Anomalia (Isolation Forest)",
    "provider_risk_score": "Risco do Prestador",
    "member_risk_score":   "Risco do Beneficiário",
    "cost_outlier_score":  "Custo Atípico",
    "risk_score":          "Score Composto",
    "claim_amount":        "Valor do Claim",
    "flag_count":          "Nº de Flags",
}


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from a scored claims DataFrame."""
    feats = pd.DataFrame(index=df.index)

    for col in ["anomaly_score", "provider_risk_score", "member_risk_score",
                "cost_outlier_score", "risk_score", "claim_amount"]:
        feats[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Count how many distinct flags the claim has
    if "risk_flags" in df.columns:
        feats["flag_count"] = df["risk_flags"].apply(
            lambda f: 0 if str(f).strip() in
                      ("Sem sinais especificos detectados", "Sem sinais específicos detectados", "")
                      else len(str(f).split(";"))
        )
    else:
        feats["flag_count"] = 0

    return feats[FEATURE_COLS]


# ── Training ──────────────────────────────────────────────────────────────────

def train(scored_df: pd.DataFrame, feedback_df: pd.DataFrame) -> tuple:
    """
    Train the supervised classifier using investigator feedback as labels.

    Parameters
    ----------
    scored_df   : full scored claims DataFrame (output of risk_scorer.compute)
    feedback_df : feedback rows from data_loader.load_feedback()

    Returns
    -------
    (model, metrics_dict)
    model is None if there is insufficient labeled data.
    metrics_dict always describes the outcome.
    """
    if feedback_df is None or feedback_df.empty:
        return None, {"status": "no_feedback",
                      "message": "Nenhum feedback de investigadores registado ainda."}

    # Keep only confirmed verdicts — skip "Em Investigação" (uncertain)
    labeled = feedback_df[
        feedback_df["verdict"].isin(["Fraude Confirmada", "Falso Positivo"])
    ].copy()

    n_fraud  = (labeled["verdict"] == "Fraude Confirmada").sum()
    n_legit  = (labeled["verdict"] == "Falso Positivo").sum()

    if n_fraud < MIN_SAMPLES or n_legit < MIN_SAMPLES:
        return None, {
            "status": "insufficient",
            "message": (
                f"São necessários pelo menos {MIN_SAMPLES} exemplos de cada classe. "
                f"Actuais: {n_fraud} Fraude Confirmada, {n_legit} Falso Positivo."
            ),
            "n_fraud": int(n_fraud),
            "n_legit": int(n_legit),
            "needed":  MIN_SAMPLES,
        }

    # Merge labels with scored features
    merged = scored_df.merge(
        labeled[["claim_id", "verdict"]], on="claim_id", how="inner"
    )

    if len(merged) < MIN_SAMPLES * 2:
        return None, {
            "status": "insufficient",
            "message": "Não há sobreposição suficiente entre o feedback e os claims actuais.",
            "n_labeled": len(merged),
        }

    X = _extract_features(merged)
    y = (merged["verdict"] == "Fraude Confirmada").astype(int)

    # Train Random Forest — class_weight='balanced' handles imbalanced labels
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # ── Evaluation ───────────────────────────────────────────────────────────
    metrics = {
        "status":    "trained",
        "n_labeled": len(merged),
        "n_fraud":   int(n_fraud),
        "n_legit":   int(n_legit),
    }

    # Cross-validation only when we have enough samples for meaningful folds
    if len(merged) >= MIN_SAMPLES * 4:
        n_splits = min(5, int(len(merged) / (MIN_SAMPLES * 2)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        metrics["f1_cv_mean"] = round(float(f1_scores.mean()), 3)
        metrics["f1_cv_std"]  = round(float(f1_scores.std()),  3)

    # In-sample metrics (optimistic but useful for the dashboard)
    y_pred = model.predict(X)
    metrics["precision"] = round(float(precision_score(y, y_pred, zero_division=0)), 3)
    metrics["recall"]    = round(float(recall_score(y, y_pred, zero_division=0)), 3)
    metrics["f1"]        = round(float(f1_score(y, y_pred, zero_division=0)), 3)

    # Feature importance
    metrics["feature_importance"] = {
        FEATURE_LABELS[col]: round(float(imp), 3)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }

    save_model(model, metrics)
    return model, metrics


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(scored_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Add fraud_probability and supervised_verdict columns to scored_df.

    fraud_probability : float 0–100  (probability of being real fraud)
    supervised_verdict: str  ("Provável Fraude" / "Incerto" / "Provável Legítimo")

    Falls back gracefully to None columns when no model is available.
    """
    out = scored_df.copy()

    if model is None:
        model = load_model()

    if model is None:
        out["fraud_probability"]  = None
        out["supervised_verdict"] = None
        return out

    X = _extract_features(out)
    proba = model.predict_proba(X)[:, 1]   # P(fraud)

    out["fraud_probability"] = (proba * 100).round(1)
    out["supervised_verdict"] = pd.cut(
        out["fraud_probability"],
        bins=[-0.1, 30.0, 65.0, 100.1],
        labels=["Provável Legítimo", "Incerto", "Provável Fraude"],
    ).astype(str)

    return out


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: dump to a temp file then rename, so an interrupted
    # training run can never leave a corrupt half-written pickle behind
    tmp_path = MODEL_PATH.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)
    os.replace(tmp_path, MODEL_PATH)


def load_model():
    """Return trained model or None if not yet trained."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data.get("model")
    except Exception:
        return None


def load_metrics() -> dict:
    """Return saved training metrics dict or None."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data.get("metrics")
    except Exception:
        return None


def model_is_trained() -> bool:
    return MODEL_PATH.exists()


# ── SHAP explainability ───────────────────────────────────────────────────────

def explain_claim(claim_row: pd.Series, model=None) -> dict | None:
    """
    Return a SHAP waterfall dict for a single claim.

    Uses TreeExplainer (exact, fast) — no sampling approximation.

    Returns
    -------
    dict with keys:
        "shap_values"   : list[float]  — per-feature SHAP values
        "base_value"    : float        — model expected value (log-odds)
        "feature_names" : list[str]    — human-readable labels
        "feature_values": list[float]  — raw feature values for the claim
    or None if shap is unavailable or model not trained.
    """
    try:
        import shap  # optional — graceful fallback if not installed
    except ImportError:
        return None

    if model is None:
        model = load_model()
    if model is None:
        return None

    X = _extract_features(claim_row.to_frame().T)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # shap_values for binary classifier: list[2 arrays]; index 1 = P(fraud)
    if isinstance(sv, list):
        shap_vals = sv[1][0]
    else:
        shap_vals = sv[0]

    return {
        "shap_values":    shap_vals.tolist(),
        "base_value":     float(explainer.expected_value[1]
                                if isinstance(explainer.expected_value, (list, np.ndarray))
                                else explainer.expected_value),
        "feature_names":  [FEATURE_LABELS[c] for c in FEATURE_COLS],
        "feature_values": X.iloc[0].tolist(),
    }
