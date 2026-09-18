import json
import os
import time
import boto3
# from ulid import ULID
from ulid import ULID
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['Ingestion_Table'])

def lambda_handler(event, context):
    body = json.loads(event['body'])

    sender_id = body['senderId']
    receiver_id = body['receiverId']
    amount = str(body['amount'])
    currency = body['currency']

    # txn_id = str(ulid.new())
    txn_id = str(ULID())
    timestamp = str(int(time.time()*1000))

    # sorted_pair = "-".json(sorted([sender_id, receiver_id]))
    sorted_pair = "-".join(sorted([sender_id, receiver_id]))
    with table.batch_writer() as batch:
        # 1. Canonical Transaction Record
        batch.put_item(Item={
            'PK': f'TXN#{txn_id}',
            'SK': 'META',
            'senderId': sender_id,
            'receiverId': receiver_id,
            'amount': amount,
            'currency': currency,
            'timestamp': timestamp,
            'status': 'PENDING'
        })

        batch.put_item(Item={
            'PK': f'ACCOUNT#{sender_id}',
            'SK': f'EDGE#{timestamp}#{txn_id}',
            'direction': 'OUT',
            'counterpartyId': receiver_id,
            'amount': amount,
            'GSI2PK': f'PAIR#{sorted_pair}',
            'GSI2SK': timestamp
        })

        batch.put_item(Item={
            'PK': f'ACCOUNT#{receiver_id}',
            'SK': f'EDGE#{timestamp}#{txn_id}',
            'direction': 'IN',
            'counterpartyId': sender_id,
            'amount': amount
        })

        return {
            "statusCode": 202,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "message": "Transaction ingested",
                "txnId": txn_id
            })
        }