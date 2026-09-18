import json

# Scoring weights approximating a GraphSAGE aggregator
WEIGHT_STRUCTURING = 0.45
WEIGHT_VELOCITY = 0.25
WEIGHT_DIVERSITY = 0.20
WEIGHT_AMOUNT_DEVIATION = 0.10

ANOMALY_THRESHOLD = 0.70

def calculate_anomaly_score(features, current_amount):
    # 1. Structuring Heuristic: Rapid transactions between $9,000 and $10,000
    structuring_count = features.get("structuring_cluster_count", 0)
    structuring_score = min(structuring_count / 5.0, 1.0)

    # 2. Velocity / Pair Density
    pair_count = features.get("pair_tx_count", 0)
    velocity_score = min(pair_count / 10.0, 1.0)

    # 3. Counterparty Concentration (Low diversity = higher risk for high volume)
    diversity = features.get("sender_counterparty_diversity", 1)
    out_degree = features.get("sender_out_degree", 1)
    if out_degree > 3:
        diversity_score = 1.0 - (diversity / out_degree)
    else:
        diversity_score = 0.1

    # 4. Amount Deviation from Sender Average
    avg_out = features.get("sender_avg_out_amount", current_amount)
    if avg_out > 0:
        amount_ratio = current_amount / avg_out
        amount_score = min(max((amount_ratio - 1.0) / 2.0, 0.0), 1.0)
    else:
        amount_score = 0.0

    # Composite weighted score [0.0, 1.0]
    final_score = (
        (structuring_score * WEIGHT_STRUCTURING) +
        (velocity_score * WEIGHT_VELOCITY) +
        (diversity_score * WEIGHT_DIVERSITY) +
        (amount_score * WEIGHT_AMOUNT_DEVIATION)
    )
    return round(final_score, 4)


def lambda_handler(event, context):
    """
    Accepts output from FeatureExtractionFunction:
    { "txnId": "...", "senderId": "...", "receiverId": "...", "currentAmount": 9850.0, "features": { ... } }
    """
    features = event.get("features", {})
    current_amount = float(event.get("currentAmount", 0.0))

    score = calculate_anomaly_score(features, current_amount)
    flagged = score >= ANOMALY_THRESHOLD

    subgraph_summary = (
        f"Sender {event['senderId']} has {features.get('sender_out_degree', 0)} out-edges "
        f"with {features.get('sender_counterparty_diversity', 0)} unique counterparty. "
        f"Pair history shows {features.get('pair_tx_count', 0)} transactions totaling "
        f"${features.get('pair_total_volume', 0):,.2f}, with {features.get('structuring_cluster_count', 0)} "
        f"structuring transfers clustered below $10,000."
    )

    return {
        "txnId": event.get("txnId"),
        "senderId": event.get("senderId"),
        "receiverId": event.get("receiverId"),
        "currentAmount": current_amount,
        "features": features,
        "score": score,
        "threshold": ANOMALY_THRESHOLD,
        "flagged": flagged,
        "subgraph_summary": subgraph_summary
    }