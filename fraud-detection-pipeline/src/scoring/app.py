"""
Graph Anomaly Scoring Lambda — Hybrid ML + Graph Ensemble
==========================================================
Combines a trained XGBoost model (PaySim) with graph-based heuristics
to produce a final anomaly score for each transaction.

Model artifacts are loaded from a Lambda layer at /opt/ml/model/.
Falls back to pure graph heuristics if the ML model is unavailable.
"""

import json
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Graph-based heuristic scoring (original — kept as ensemble component)
# ---------------------------------------------------------------------------
WEIGHT_STRUCTURING = 0.45
WEIGHT_VELOCITY = 0.25
WEIGHT_DIVERSITY = 0.20
WEIGHT_AMOUNT_DEVIATION = 0.10

ANOMALY_THRESHOLD = 0.70


def calculate_graph_score(features, current_amount):
    """Original rule-based graph scoring heuristic."""
    structuring_count = features.get("structuring_cluster_count", 0)
    structuring_score = min(structuring_count / 5.0, 1.0)

    pair_count = features.get("pair_tx_count", 0)
    velocity_score = min(pair_count / 10.0, 1.0)

    diversity = features.get("sender_counterparty_diversity", 1)
    out_degree = features.get("sender_out_degree", 1)
    if out_degree > 3:
        diversity_score = 1.0 - (diversity / out_degree)
    else:
        diversity_score = 0.1

    avg_out = features.get("sender_avg_out_amount", current_amount)
    if avg_out > 0:
        amount_ratio = current_amount / avg_out
        amount_score = min(max((amount_ratio - 1.0) / 2.0, 0.0), 1.0)
    else:
        amount_score = 0.0

    final_score = (
        (structuring_score * WEIGHT_STRUCTURING)
        + (velocity_score * WEIGHT_VELOCITY)
        + (diversity_score * WEIGHT_DIVERSITY)
        + (amount_score * WEIGHT_AMOUNT_DEVIATION)
    )
    return round(final_score, 4)


# ---------------------------------------------------------------------------
# ML Model loading (cold-start, cached across invocations)
# ---------------------------------------------------------------------------
ML_MODEL = None
FEATURE_CONFIG = None
ML_AVAILABLE = False

MODEL_JSON_PATH = os.environ.get("MODEL_JSON_PATH", "/opt/ml/model/fraud_model.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/fraud_model.joblib")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/opt/ml/model/feature_config.json")


def load_ml_model():
    """Load the XGBoost model and feature config from the Lambda layer."""
    global ML_MODEL, FEATURE_CONFIG, ML_AVAILABLE
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                FEATURE_CONFIG = json.load(f)

        # 1. Prefer native Booster JSON format (lightweight: only needs xgboost + numpy)
        if os.path.exists(MODEL_JSON_PATH):
            # Stub scipy if not present to avoid bundling 150MB of unused scipy wheels
            import sys
            import types
            if "scipy" not in sys.modules:
                scipy = types.ModuleType("scipy")
                scipy.__path__ = []
                scipy.sparse = types.ModuleType("scipy.sparse")
                class _StubMatrix: pass
                scipy.sparse.csr_matrix = _StubMatrix
                scipy.sparse.spmatrix = _StubMatrix
                scipy.special = types.ModuleType("scipy.special")
                scipy.special.softmax = lambda x: x
                sys.modules["scipy"] = scipy
                sys.modules["scipy.sparse"] = scipy.sparse
                sys.modules["scipy.special"] = scipy.special

            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(MODEL_JSON_PATH)
            ML_MODEL = booster
            ML_AVAILABLE = True
            logger.info("ML Booster loaded successfully from JSON")
        # 2. Fall back to joblib if json not present
        elif os.path.exists(MODEL_PATH):
            import joblib
            ML_MODEL = joblib.load(MODEL_PATH)
            ML_AVAILABLE = True
            logger.info("ML model loaded successfully from joblib")
        else:
            logger.warning(
                f"ML model artifacts not found at {MODEL_JSON_PATH} or {MODEL_PATH} — using graph heuristics only"
            )
    except Exception as e:
        logger.error(f"Failed to load ML model: {e} — falling back to graph heuristics")
        ML_AVAILABLE = False


# Attempt model load on cold start
load_ml_model()


def calculate_ml_score(features, current_amount):
    """
    Build a feature vector matching the training schema and run XGBoost inference.

    PaySim core features (balance-related) are set to neutral defaults since
    we don't track account balances in DynamoDB. The model still benefits from
    the graph-bridge features and amount-based signals.
    """
    import numpy as np
    import xgboost as xgb

    type_mapping = FEATURE_CONFIG.get("type_mapping", {}) if FEATURE_CONFIG else {}
    # Default to TRANSFER type (most common for fraud in PaySim)
    txn_type_encoded = features.get(
        "transaction_type_numeric", type_mapping.get("TRANSFER", 4)
    )

    old_orig = float(features.get("oldbalanceOrg", 0.0))
    new_orig = float(features.get("newbalanceOrig", 0.0))
    old_dest = float(features.get("oldbalanceDest", 0.0))
    new_dest = float(features.get("newbalanceDest", 0.0))

    # Derived PaySim balance features
    balance_delta_orig = old_orig - new_orig
    balance_delta_dest = new_dest - old_dest
    error_balance_orig = new_orig + current_amount - old_orig if old_orig > 0 else 0.0
    error_balance_dest = old_dest + current_amount - new_dest if new_dest > 0 else 0.0
    amount_ratio_orig = current_amount / (old_orig + 1.0) if old_orig > 0 else features.get("amount_to_avg_ratio", 0.0)
    amount_ratio_dest = current_amount / (old_dest + 1.0) if old_dest > 0 else 0.0
    is_orig_zero = 1 if (old_orig > 0 and new_orig == 0.0) else 0
    is_dest_zero = 1 if old_dest == 0.0 else 0

    # Build feature vector in the EXACT order the model was trained on
    feature_vector = [
        # PaySim core features
        txn_type_encoded,                           # type_encoded
        current_amount,                             # amount
        old_orig,                                   # oldbalanceOrg
        new_orig,                                   # newbalanceOrig
        old_dest,                                   # oldbalanceDest
        new_dest,                                   # newbalanceDest
        # Engineered features
        balance_delta_orig,                         # balance_delta_orig
        balance_delta_dest,                         # balance_delta_dest
        error_balance_orig,                         # error_balance_orig
        error_balance_dest,                         # error_balance_dest
        amount_ratio_orig,                          # amount_ratio_orig
        amount_ratio_dest,                          # amount_ratio_dest
        is_orig_zero,                               # is_orig_balance_zero
        is_dest_zero,                               # is_dest_balance_zero
        12,                                         # hour_of_day (default midday)
        15,                                         # day_of_month (default mid-month)
        # Graph-bridge features — populated from DynamoDB graph
        features.get("sender_out_degree", 0),
        features.get("sender_avg_out_amount", 0),
        features.get("sender_counterparty_diversity", 1),
        features.get("receiver_in_degree", 0),
        features.get("receiver_avg_in_amount", 0),
        features.get("pair_tx_count", 0),
        features.get("pair_total_volume", 0),
        features.get("structuring_cluster_count", 0),
    ]

    X = np.array([feature_vector], dtype=np.float32)

    if isinstance(ML_MODEL, xgb.Booster):
        dmatrix = xgb.DMatrix(X)
        proba = float(ML_MODEL.predict(dmatrix)[0])
    else:
        proba = float(ML_MODEL.predict_proba(X)[0][1])

    return round(proba, 4)


# ---------------------------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    """
    Accepts output from FeatureExtractionFunction:
    { "txnId": "...", "senderId": "...", "receiverId": "...",
      "currentAmount": 9850.0, "features": { ... } }
    """
    features = event.get("features", {})
    current_amount = float(event.get("currentAmount", 0.0))

    # Always compute graph score
    graph_score = calculate_graph_score(features, current_amount)

    # Compute ML score if model is available
    if ML_AVAILABLE:
        try:
            ml_score = calculate_ml_score(features, current_amount)
            ml_weight = FEATURE_CONFIG.get("ensemble_weight_ml", 0.70)
            graph_weight = FEATURE_CONFIG.get("ensemble_weight_graph", 0.30)
            final_score = round(ml_weight * ml_score + graph_weight * graph_score, 4)
            scoring_method = "ensemble"
        except Exception as e:
            logger.error(f"ML scoring failed: {e} — using graph heuristic")
            final_score = graph_score
            ml_score = None
            scoring_method = "graph_fallback"
    else:
        final_score = graph_score
        ml_score = None
        scoring_method = "graph_only"

    # Flag if:
    # 1. Composite score exceeds anomaly threshold, OR
    # 2. ML model specifically detects an extreme fraud probability (>= 0.50), OR
    # 3. Graph heuristic alone detects high structuring anomaly (>= 0.70)
    ml_flagged = (ml_score is not None and ml_score >= 0.50)
    graph_flagged = graph_score >= ANOMALY_THRESHOLD
    composite_flagged = final_score >= ANOMALY_THRESHOLD

    flagged = composite_flagged or ml_flagged or graph_flagged

    subgraph_summary = (
        f"Sender {event['senderId']} has {features.get('sender_out_degree', 0)} out-edges "
        f"with {features.get('sender_counterparty_diversity', 0)} unique counterparty. "
        f"Pair history shows {features.get('pair_tx_count', 0)} transactions totaling "
        f"${features.get('pair_total_volume', 0):,.2f}, with {features.get('structuring_cluster_count', 0)} "
        f"structuring transfers clustered below $10,000."
    )

    result = {
        "txnId": event.get("txnId"),
        "senderId": event.get("senderId"),
        "receiverId": event.get("receiverId"),
        "currentAmount": current_amount,
        "features": features,
        "score": final_score,
        "threshold": ANOMALY_THRESHOLD,
        "flagged": flagged,
        "subgraph_summary": subgraph_summary,
        "scoring_method": scoring_method,
    }

    # Include detailed scores for observability
    if ml_score is not None:
        result["ml_score"] = ml_score
        result["graph_score"] = graph_score

    return result