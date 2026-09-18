import boto3
import random
import time
from ulid import ULID

# Initialize DynamoDB using your local AWS profile (azDev)
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('TransactionGraph')

USERS = [f"acc_user_{i:03d}" for i in range(1, 21)]
MERCHANTS = [f"acc_merchant_{i:03d}" for i in range(1, 6)]


def create_transaction_items(sender, receiver, amount, is_fraud=False):
    txn_id = str(ULID())
    # Backdate timestamps slightly so they exist in the past
    timestamp = str(int(time.time() * 1000) - random.randint(1000, 86400000))
    amount_str = f"{amount:.2f}"

    sorted_pair = "-".join(sorted([sender, receiver]))

    # 1. Canonical Record
    canonical = {
        "PK": f"TXN#{txn_id}",
        "SK": "META",
        "senderId": sender,
        "receiverId": receiver,
        "amount": amount_str,
        "currency": "USD",
        "timestamp": timestamp,
        "status": "COMPLETED",
        "is_synthetic_fraud": is_fraud  # Helpful flag for your demo
    }

    # 2. Sender Outbound Edge
    outbound = {
        "PK": f"ACCOUNT#{sender}",
        "SK": f"EDGE#{timestamp}#{txn_id}",
        "counterpartyId": receiver,
        "direction": "OUT",
        "amount": amount_str,
        "GSI2PK": f"PAIR#{sorted_pair}",
        "GSI2SK": timestamp
    }

    # 3. Receiver Inbound Edge
    inbound = {
        "PK": f"ACCOUNT#{receiver}",
        "SK": f"EDGE#{timestamp}#{txn_id}",
        "counterpartyId": sender,
        "direction": "IN",
        "amount": amount_str
    }

    return [canonical, outbound, inbound]


def seed_database():
    print("Seeding DynamoDB...")
    items_to_write = []

    # 1. Generate Normal Background Traffic
    for _ in range(50):
        sender = random.choice(USERS)
        receiver = random.choice(MERCHANTS)
        amount = random.uniform(10.0, 500.0)
        items_to_write.extend(create_transaction_items(sender, receiver, amount))

    # 2. Generate Structuring / Smurfing Fraud Pattern
    # A single user sending many transfers just under a $10,000 reporting threshold to the same counterparty
    fraud_sender = "acc_user_999_FRAUD"
    fraud_receiver = "acc_merchant_999_FRAUD"
    for _ in range(12):
        amount = random.uniform(9500.0, 9950.0)
        items_to_write.extend(create_transaction_items(fraud_sender, fraud_receiver, amount, is_fraud=True))

    # Batch write items to DynamoDB (Max 25 per batch)
    with table.batch_writer() as batch:
        for item in items_to_write:
            batch.put_item(Item=item)

    print(f"Successfully seeded {len(items_to_write) // 3} transactions ({(len(items_to_write))} total edges/records).")


if __name__ == "__main__":
    seed_database()