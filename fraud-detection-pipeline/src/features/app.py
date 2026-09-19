import os
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ.get('TABLE_NAME', 'TransactionGraph'))

def query_edges(account_id, limit=50):
    """Fetches up to limit recent edges for an account."""
    response = table.query(
        KeyConditionExpression=Key('PK').eq(f"ACCOUNT#{account_id}") & Key('SK').begins_with("EDGE#"),
        ScanIndexForward=False,  # Most recent first
        Limit=limit
    )
    return response.get('Items', [])

def query_pair_history(sender_id, receiver_id, limit=50):
    """Fetches pair-level history using PairIndex (GSI2)."""
    sorted_pair = "-".join(sorted([sender_id, receiver_id]))
    response = table.query(
        IndexName='PairIndex',
        KeyConditionExpression=Key('GSI2PK').eq(f"PAIR#{sorted_pair}"),
        ScanIndexForward=False,
        Limit=limit
    )
    return response.get('Items', [])


def lambda_handler(event, context):
    """
    Accepts transaction payload from Step Functions or Ingestion:
    { "senderId": "...", "receiverId": "...", "amount": 100.0, "txnId": "...", "timestamp": "..." }
    """
    sender_id = event['senderId']
    receiver_id = event['receiverId']
    current_amount = float(event['amount'])

    # 1-hop neighborhood queries
    sender_edges = query_edges(sender_id)
    receiver_edges = query_edges(receiver_id)
    pair_edges = query_pair_history(sender_id, receiver_id)

    # Compute sender local graph features
    outbound_amounts = [float(e['amount']) for e in sender_edges if e.get('direction') == 'OUT']
    sender_out_degree = len(outbound_amounts)
    sender_avg_out_amount = sum(outbound_amounts) / sender_out_degree if sender_out_degree > 0 else 0.0
    sender_unique_counterparties = len(set(e.get('counterpartyId') for e in sender_edges))

    # Compute receiver local graph features
    inbound_amounts = [float(e['amount']) for e in receiver_edges if e.get('direction') == 'IN']
    receiver_in_degree = len(inbound_amounts)
    receiver_avg_in_amount = sum(inbound_amounts) / receiver_in_degree if receiver_in_degree > 0 else 0.0

    # Compute pair-level structuring features (near $10k threshold)
    pair_amounts = [float(e['amount']) for e in pair_edges]
    pair_count_recent = len(pair_amounts)
    pair_total_volume = sum(pair_amounts)
    structuring_transfers = sum(1 for a in pair_amounts if 9000.0 <= a < 10000.0)

    feature_vector = {
        "txnId": event.get('txnId', ''),
        "senderId": sender_id,
        "receiverId": receiver_id,
        "currentAmount": current_amount,
        "features": {
            "sender_out_degree": sender_out_degree,
            "sender_avg_out_amount": round(sender_avg_out_amount, 2),
            "sender_counterparty_diversity": sender_unique_counterparties,
            "receiver_in_degree": receiver_in_degree,
            "receiver_avg_in_amount": round(receiver_avg_in_amount, 2),
            "pair_tx_count": pair_count_recent,
            "pair_total_volume": round(pair_total_volume, 2),
            "structuring_cluster_count": structuring_transfers,
            "amount_to_avg_ratio": round(current_amount / sender_avg_out_amount, 4) if sender_avg_out_amount > 0 else 0.0,
            "is_high_value": 1 if current_amount > 200000 else 0,
            "transaction_type_numeric": event.get("transactionTypeNumeric", 4),  # Default: TRANSFER
            "oldbalanceOrg": float(event.get("oldbalanceOrg", 0.0)),
            "newbalanceOrig": float(event.get("newbalanceOrig", 0.0)),
            "oldbalanceDest": float(event.get("oldbalanceDest", 0.0)),
            "newbalanceDest": float(event.get("newbalanceDest", 0.0)),
        }
    }

    return feature_vector