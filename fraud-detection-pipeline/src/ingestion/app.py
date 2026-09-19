import json
import os
import uuid
import time
from decimal import Decimal
import boto3

dynamodb = boto3.resource("dynamodb")
sfn_client = boto3.client("stepfunctions")

TABLE_NAME = os.environ.get("TABLE_NAME", "TransactionGraph")
STATE_MACHINE_ARN = os.environ.get("WORKFLOW_ARN")
table = dynamodb.Table(TABLE_NAME)


def persist_transaction_graph(txn_id, sender_id, receiver_id, amount, currency, timestamp, status):
    """Writes transaction record and bidirectional graph edges to DynamoDB."""
    amount_dec = Decimal(str(amount))
    sorted_pair = "-".join(sorted([sender_id, receiver_id]))

    with table.batch_writer() as batch:
        # 1. Main Transaction Record
        batch.put_item(Item={
            "PK": f"TXN#{txn_id}",
            "SK": "METADATA",
            "GSI1PK": f"STATUS#{status}",
            "GSI1SK": f"TIMESTAMP#{timestamp}",
            "senderId": sender_id,
            "receiverId": receiver_id,
            "amount": amount_dec,
            "currency": currency,
            "timestamp": timestamp,
            "status": status
        })
        # 2. Sender Outbound Edge
        batch.put_item(Item={
            "PK": f"ACCOUNT#{sender_id}",
            "SK": f"EDGE#{timestamp}#{txn_id}",
            "counterpartyId": receiver_id,
            "direction": "OUT",
            "amount": amount_dec,
            "timestamp": timestamp,
            "GSI2PK": f"PAIR#{sorted_pair}",
            "GSI2SK": str(timestamp)
        })
        # 3. Receiver Inbound Edge
        batch.put_item(Item={
            "PK": f"ACCOUNT#{receiver_id}",
            "SK": f"EDGE#{timestamp}#{txn_id}",
            "counterpartyId": sender_id,
            "direction": "IN",
            "amount": amount_dec,
            "timestamp": timestamp,
            "GSI2PK": f"PAIR#{sorted_pair}",
            "GSI2SK": str(timestamp)
        })


def lambda_handler(event, context):
    try:
        http_method = event.get("httpMethod")
        if http_method == "GET":
            return handle_get_alerts()

        raw_body = event.get("body") or "{}"
        body = json.loads(raw_body)

        txn_id = body.get("txnId", f"txn_{uuid.uuid4().hex[:12]}")
        timestamp = body.get("timestamp", int(time.time() * 1000))
        sender_id = body["senderId"]
        receiver_id = body["receiverId"]
        amount = float(body["amount"])
        currency = body.get("currency", "USD")

        execution_input = {
            "txnId": txn_id,
            "senderId": sender_id,
            "receiverId": receiver_id,
            "amount": amount,
            "currency": currency,
            "timestamp": timestamp
        }

        # 1. Run Step Functions Express Workflow
        response = sfn_client.start_sync_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input=json.dumps(execution_input)
        )

        status = response.get("status")
        if status != "SUCCEEDED":
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({
                    "error": "WorkflowExecutionFailed",
                    "status": status,
                    "cause": response.get("cause")
                })
            }

        output = json.loads(response.get("output", "{}"))
        decision = output.get("decision", {})
        decision_status = decision.get("status", "PENDING")

        # 2. Persist graph edges and transaction record
        persist_transaction_graph(
            txn_id=txn_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            currency=currency,
            timestamp=timestamp,
            status=decision_status
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "txnId": txn_id,
                "decision": decision,
                "score": output.get("score"),
                "flagged": output.get("flagged"),
                "explanation": output.get("explanation")
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
def handle_get_alerts():
    """Queries StatusIndex (GSI1) for recent flagged transactions."""
    response = table.query(
        IndexName="StatusIndex",
        KeyConditionExpression="GSI1PK = :status",
        ExpressionAttributeValues={":status": "STATUS#FLAGGED"},
        ScanIndexForward=False,
        Limit=20
    )
    items = response.get("Items", [])
    # Convert Decimals for JSON serialization
    for item in items:
        if "amount" in item:
            item["amount"] = float(item["amount"])
        if "timestamp" in item:
            item["timestamp"] = int(item["timestamp"])
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(items)
    }