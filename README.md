# 🛡️ Real-Time AML & Fraud Detection Pipeline

> A production-grade, serverless **Anti-Money Laundering (AML) & Fraud Detection** system built on AWS — powered by **DynamoDB graph modeling**, **Step Functions orchestration**, **heuristic anomaly scoring**, and **Amazon Bedrock (Nova Pro) AI explanations**.

[![AWS SAM](https://img.shields.io/badge/Built%20with-AWS%20SAM-orange?logo=amazonaws)](https://aws.amazon.com/serverless/sam/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Deployment](#setup--deployment)
- [Seed Data](#seed-data)
- [API Reference](#api-reference)
- [Pipeline Workflow](#pipeline-workflow)
- [Scoring Algorithm](#scoring-algorithm)
- [Bedrock AI Explanation](#bedrock-ai-explanation)
- [Frontend Dashboard](#frontend-dashboard)
- [CloudWatch Alarms (SRE)](#cloudwatch-alarms-sre)
- [Sample Responses](#sample-responses)
- [Cost Estimation](#cost-estimation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Financial institutions are required to detect and report suspicious transactions in real time. This pipeline ingests transactions via API Gateway, orchestrates a multi-stage analysis workflow through AWS Step Functions (Express), and produces human-readable risk explanations using Amazon Bedrock's Nova Pro foundation model.

**Core flow:** A transaction enters the system → graph features are extracted from DynamoDB → an anomaly score is computed → if flagged, Bedrock generates a compliance explanation → an SNS alert is published to the compliance team.

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────────────────────────────────────────┐
│   Frontend   │───▶│   API Gateway    │───▶│           Step Functions (Express Workflow)              │
│  Dashboard   │    │  /transactions   │    │                                                         │
└──────────────┘    │  /alerts (GET)   │    │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
                    └──────────────────┘    │  │  Feature     │  │  Graph       │  │  Bedrock      │   │
                                           │  │  Extraction  │─▶│  Anomaly     │─▶│  Explanation  │   │
                                           │  │  Lambda      │  │  Scoring     │  │  Lambda       │   │
                                           │  └─────────────┘  └──────────────┘  └───────┬───────┘   │
                                           │         │                                    │           │
                                           └─────────┼────────────────────────────────────┼───────────┘
                                                     │                                    │
                                                     ▼                                    ▼
                                           ┌──────────────────┐                 ┌──────────────────┐
                                           │    DynamoDB       │                 │   SNS Topic      │
                                           │  TransactionGraph │                 │  AML-Compliance  │
                                           │  (Single Table)   │                 │  -Alerts         │
                                           └──────────────────┘                 └──────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Graph Data Model** | DynamoDB single-table design with bidirectional edges (sender→receiver, receiver→sender) and pair-level GSI for structuring detection |
| **Real-Time Scoring** | Sub-second anomaly scoring via Step Functions Express synchronous execution |
| **Structuring / Smurfing Detection** | Detects patterns of transactions clustered just below the \$10,000 CTR threshold |
| **AI Explanations** | Amazon Bedrock (Nova Pro) generates compliance-ready risk narratives with structured JSON output |
| **SNS Compliance Alerts** | Flagged transactions trigger immediate SNS notifications to compliance officers |
| **SRE Hardening** | CloudWatch alarms for API 5XX errors and Step Function execution failures |
| **Frontend Console** | Dark-themed monitoring dashboard with transaction simulator and live alert feed |
| **Seed Data Generator** | Script to populate DynamoDB with realistic normal + fraudulent transaction patterns |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **IaC** | AWS SAM (CloudFormation) |
| **Runtime** | Python 3.11 |
| **API** | Amazon API Gateway (REST, with request validation) |
| **Orchestration** | AWS Step Functions (Express / Synchronous) |
| **Database** | Amazon DynamoDB (Single-Table Design, Streams, GSIs) |
| **AI/ML** | Amazon Bedrock — `amazon.nova-pro-v1:0` |
| **Notifications** | Amazon SNS |
| **Monitoring** | Amazon CloudWatch Alarms |
| **Tracing** | AWS X-Ray (Active Tracing) |
| **Frontend** | Vanilla HTML/CSS/JS (no framework) |

---

## Project Structure

```
fraud-detection-pipeline/
├── README.md
└── fraud-detection-pipeline/
    ├── template.yml                          # SAM CloudFormation template (all resources)
    ├── samconfig.toml                        # SAM CLI deployment config
    ├── seed_data.py                          # Populate DynamoDB with test data
    │
    ├── src/
    │   ├── ingestion/                        # Lambda: API ingestion + alerts endpoint
    │   │   ├── app.py                        #   POST /transactions & GET /alerts
    │   │   └── requirements.txt              #   python-ulid, boto3
    │   │
    │   ├── features/                         # Lambda: Graph feature extraction
    │   │   └── app.py                        #   1-hop neighborhood + pair queries
    │   │
    │   ├── scoring/                          # Lambda: Heuristic anomaly scoring
    │   │   └── app.py                        #   Weighted composite score [0.0–1.0]
    │   │
    │   └── explanation/                      # Lambda: Bedrock AI explanation
    │       └── app.py                        #   Nova Pro structured risk narrative
    │
    ├── statemachine/
    │   └── fraud_detection_workflow.asl.json  # Step Functions ASL definition
    │
    ├── frontend/
    │   └── index.html                        # Full monitoring console + transaction simulator
    │
    ├── dashboard/
    │   └── index.html                        # Minimal alert stream dashboard
    │
    ├── response.json                         # Sample: clean transaction response
    ├── scored_response.json                  # Sample: scored transaction response
    └── explained_response.json               # Sample: flagged + explained response
```

---

## Prerequisites

- **AWS Account** with appropriate permissions (Lambda, DynamoDB, Step Functions, API Gateway, SNS, Bedrock, CloudWatch)
- **AWS CLI** configured with credentials (`aws configure`)
- **AWS SAM CLI** ≥ 1.90.0 — [Install Guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- **Python 3.11+**
- **Amazon Bedrock Model Access** — Enable `amazon.nova-pro-v1:0` in the [Bedrock Console](https://console.aws.amazon.com/bedrock/home#/modelaccess) for `us-east-1`

---

## Setup & Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/fraud-detection-pipeline.git
cd fraud-detection-pipeline/fraud-detection-pipeline
```

### 2. Build the SAM Application

```bash
sam build
```

### 3. Deploy to AWS

```bash
sam deploy --guided
```

During guided deployment, you'll be prompted for:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Stack Name | `fraud-detection-pipeline` | CloudFormation stack name |
| AWS Region | `us-east-1` | Deployment region |
| Confirm changeset | `true` | Review before deploying |
| Allow SAM CLI IAM role creation | `Yes` | Required for Lambda execution roles |

### 4. Note the Outputs

After deployment, SAM outputs three key values:

```
-----------------------------------------------------------
Outputs
-----------------------------------------------------------
Key                 TransactionApiUrl
Description         API Gateway endpoint URL for Prod stage
Value               https://<api-id>.execute-api.us-east-1.amazonaws.com/Prod/transactions

Key                 AlertsApiUrl
Description         Endpoint URL to fetch recent flagged transactions
Value               https://<api-id>.execute-api.us-east-1.amazonaws.com/Prod/alerts

Key                 ComplianceTopicArn
Description         ARN for the SNS Compliance Alerts Topic
Value               arn:aws:sns:us-east-1:<account-id>:AML-Compliance-Alerts
-----------------------------------------------------------
```

### 5. Subscribe to SNS Alerts (Optional)

```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:<account-id>:AML-Compliance-Alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

---

## Seed Data

Populate the `TransactionGraph` DynamoDB table with realistic test data:

```bash
pip install python-ulid boto3
python seed_data.py
```

This generates:

- **50 normal transactions** — random users → merchants (\$10–\$500)
- **12 fraudulent structuring transactions** — single user sending \$9,500–\$9,950 to one merchant (smurfing pattern)

Each transaction creates **3 DynamoDB items**: canonical record + sender edge + receiver edge = **186 total items**.

---

## API Reference

### POST `/transactions`

Submit a new transaction for real-time fraud analysis.

**Request Body:**

```json
{
  "senderId": "acc_user_001",
  "receiverId": "acc_merchant_001",
  "amount": 250.00,
  "currency": "USD"
}
```

**Response (Clean Transaction — APPROVED):**

```json
{
  "txnId": "txn_a1b2c3d4e5f6",
  "decision": {
    "status": "APPROVED",
    "action": "NONE"
  },
  "score": 0.15,
  "flagged": false,
  "explanation": null
}
```

**Response (Flagged Transaction — FLAGGED):**

```json
{
  "txnId": "txn_f6e5d4c3b2a1",
  "decision": {
    "status": "FLAGGED",
    "action": "HOLD_FOR_COMPLIANCE"
  },
  "score": 0.8923,
  "flagged": true,
  "explanation": {
    "risk_level": "HIGH",
    "summary": "Sender acc_user_999_FRAUD has 24 out-edges with 1 unique counterparty. Pair history shows 24 transactions totaling $233,525.92, with 24 structuring transfers clustered below $10,000.",
    "key_indicators": [
      {
        "feature": "score",
        "value": "0.8923",
        "concern": "Exceeded rule threshold"
      },
      {
        "feature": "structuring_cluster_count",
        "value": "24",
        "concern": "Multiple near-threshold transactions"
      }
    ],
    "recommended_action": "REVIEW",
    "confidence": "MEDIUM"
  }
}
```

### GET `/alerts`

Retrieve the 20 most recent flagged transactions from the `StatusIndex` GSI.

**Response:**

```json
[
  {
    "PK": "TXN#txn_a1b2c3d4e5f6",
    "SK": "METADATA",
    "senderId": "acc_user_999_FRAUD",
    "receiverId": "acc_merchant_999_FRAUD",
    "amount": 9850.00,
    "currency": "USD",
    "timestamp": 1726700000000,
    "status": "FLAGGED"
  }
]
```

---

## Pipeline Workflow

The Step Functions Express Workflow executes **synchronously** with the following state machine:

```
                    ┌─────────────────┐
                    │ ExtractFeatures  │
                    │ (Lambda)         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  ScoreAnomaly   │
                    │  (Lambda)        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  CheckFlagged   │
                    │  (Choice)        │
                    └───┬─────────┬───┘
                        │         │
                flagged=true  flagged=false
                        │         │
                        ▼         ▼
           ┌──────────────┐  ┌──────────────────┐
           │  Generate     │  │  Approve          │
           │  Explanation  │  │  Transaction      │
           │  (Bedrock)    │  │  (Pass → END)     │
           └──────┬───────┘  └──────────────────┘
                  │
                  ▼
           ┌──────────────┐
           │  Publish SNS  │
           │  Alert         │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │  Flag For     │
           │  Review       │
           │  (Pass → END) │
           └──────────────┘
```

**Retry Policy:** Each Lambda task retries up to 2 times on service exceptions with exponential backoff (1s base, 2× rate).

---

## Scoring Algorithm

The `GraphAnomalyScoringFunction` computes a **weighted composite anomaly score** in the range `[0.0, 1.0]`:

```
Score = (Structuring × 0.45) + (Velocity × 0.25) + (Diversity × 0.20) + (Amount Deviation × 0.10)
```

| Component | Weight | Calculation | Detects |
|-----------|--------|-------------|---------|
| **Structuring** | 45% | `min(structuring_cluster_count / 5, 1.0)` | Transactions clustered \$9,000–\$10,000 (smurfing) |
| **Velocity** | 25% | `min(pair_tx_count / 10, 1.0)` | Rapid transaction frequency between same pair |
| **Diversity** | 20% | `1.0 - (unique_counterparties / out_degree)` | Low counterparty diversity (concentration risk) |
| **Amount Deviation** | 10% | `min(max((amount/avg - 1) / 2, 0), 1.0)` | Unusual transaction amounts vs. sender history |

**Threshold:** Transactions scoring **≥ 0.70** are flagged for compliance review.

---

## Bedrock AI Explanation

When a transaction is flagged (`score ≥ 0.70`), the **BedrockExplanationFunction** invokes **Amazon Nova Pro** (`amazon.nova-pro-v1:0`) to generate a structured compliance explanation.

### Output Schema

```json
{
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "summary": "2-4 sentence compliance-officer explanation",
  "key_indicators": [
    {
      "feature": "structuring_cluster_count",
      "value": "24",
      "concern": "Multiple near-threshold transactions"
    }
  ],
  "recommended_action": "MONITOR | REVIEW | FREEZE_AND_ESCALATE",
  "confidence": "LOW | MEDIUM | HIGH"
}
```

### Fallback Behavior

If Bedrock is unavailable or returns an error, the Lambda generates a **deterministic fallback explanation** based on the computed score and features — ensuring the pipeline never fails silently.

---

## Frontend Dashboard

### Monitoring Console (`frontend/index.html`)

A dark-themed, real-time AML monitoring console featuring:

- **Transaction Simulator** — Submit clean or fraudulent (structuring pattern) transactions directly from the browser
- **Pipeline Response Viewer** — See the full scoring decision, anomaly score, and Bedrock explanation inline
- **Flagged Audit Stream** — Live feed of flagged transactions pulled from the `StatusIndex` GSI

**To use:** Open `frontend/index.html` in your browser and update the `API_URL` and `ALERTS_URL` constants with your deployed API Gateway endpoints.

### Minimal Dashboard (`dashboard/index.html`)

A lightweight alert stream page for embedding or extending.

---

## CloudWatch Alarms (SRE)

The template provisions two production-hardening alarms:

| Alarm | Metric | Threshold | Period |
|-------|--------|-----------|--------|
| `Api5xxAlarm` | `AWS/ApiGateway` → `5XXError` | ≥ 5 errors | 60 seconds |
| `WorkflowFailureAlarm` | `AWS/States` → `ExecutionsFailed` | ≥ 1 failure | 60 seconds |

Connect these to an SNS topic or PagerDuty for on-call alerting.

---

## DynamoDB Single-Table Design

### Access Patterns

| Access Pattern | Key Condition | Index |
|---|---|---|
| Get transaction by ID | `PK = TXN#<txnId>`, `SK = METADATA` | Table |
| Get account edges (1-hop) | `PK = ACCOUNT#<id>`, `SK begins_with EDGE#` | Table |
| List flagged transactions | `GSI1PK = STATUS#FLAGGED` | StatusIndex (GSI1) |
| Pair transaction history | `GSI2PK = PAIR#<sorted_pair>` | PairIndex (GSI2) |

### Item Types

```
┌──────────────────────────┬──────────────────────┬──────────────────┐
│          PK              │         SK           │     Type         │
├──────────────────────────┼──────────────────────┼──────────────────┤
│ TXN#<txnId>              │ METADATA             │ Canonical Record │
│ ACCOUNT#<senderId>       │ EDGE#<ts>#<txnId>    │ Outbound Edge    │
│ ACCOUNT#<receiverId>     │ EDGE#<ts>#<txnId>    │ Inbound Edge     │
└──────────────────────────┴──────────────────────┴──────────────────┘
```

---

## Sample Responses

The repository includes sample JSON responses at each pipeline stage for reference:

| File | Description |
|------|-------------|
| `response.json` | Raw ingestion response |
| `scored_response.json` | After feature extraction + anomaly scoring |
| `explained_response.json` | Full pipeline output with Bedrock explanation |

---

## Cost Estimation

This architecture is designed to be cost-efficient for demo and low-to-moderate traffic:

| Service | Pricing Model | Estimate (1K txns/day) |
|---------|---------------|----------------------|
| API Gateway | \$3.50 / million requests | ~\$0.11/month |
| Lambda | \$0.20 / million invocations | ~\$0.24/month |
| DynamoDB | On-demand, \$1.25 / million writes | ~\$0.11/month |
| Step Functions Express | \$1.00 / million transitions | ~\$0.15/month |
| Bedrock (Nova Pro) | Per-token pricing | ~\$2–5/month (flagged only) |
| SNS | \$0.50 / million publishes | ~\$0.01/month |

> **Total estimated cost:** **< \$10/month** for typical development/demo usage.

---

## Cleanup

To delete all deployed AWS resources:

```bash
sam delete --stack-name fraud-detection-pipeline
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using AWS Serverless
</p>
