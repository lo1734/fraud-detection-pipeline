import json
import os
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

SYSTEM_PROMPT = """You are an AML/Fraud Compliance Assistant generating explanations for a real-time transaction monitoring system. You will be given a transaction's structured feature vector and an anomaly score already computed by an upstream model.

Rules:
- Base your explanation ONLY on the provided features and score. Never invent transaction details, account history, or regulatory citations not present in the input.
- Be concise: 2-4 sentences for the summary, plain compliance-officer language, no hedging filler ("it seems", "possibly").
- Identify the SPECIFIC indicators that drove the score (e.g. structuring pattern, high counterparty diversity, velocity spike) — cite feature values.
- Output STRICT JSON matching the schema below. No markdown, no prose outside the JSON, no trailing commentary.

Output schema:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "summary": string,
  "key_indicators": [ { "feature": string, "value": string, "concern": string } ],
  "recommended_action": "MONITOR" | "REVIEW" | "FREEZE_AND_ESCALATE",
  "confidence": "LOW" | "MEDIUM" | "HIGH"
}"""


def lambda_handler(event, context):
    txn_id = event.get("txnId", "UNKNOWN")
    score = event.get("score", 0.0)
    threshold = event.get("threshold", 0.7)
    features = event.get("features", {})
    subgraph_summary = event.get("subgraph_summary", "")

    user_message = (
        f"Transaction ID: {txn_id}\n"
        f"Anomaly score: {score} (threshold: {threshold})\n"
        f"Feature vector: {json.dumps(features)}\n"
        f"Subgraph summary: {subgraph_summary}"
    )

    # Amazon Nova message payload format
    payload = {
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [
            {"role": "user", "content": [{"text": user_message}]}
        ],
        "inferenceConfig": {
            "max_new_tokens": 400,
            "temperature": 0.0
        }
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        body = json.loads(response["body"].read())

        # Amazon Nova outputs inside body['output']['message']['content'][0]['text']
        raw_output = body["output"]["message"]["content"][0]["text"].strip()

        # Strip markdown fences if the model wraps the JSON
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()

        explanation = json.loads(raw_output)
    except Exception as e:
        explanation = {
            "risk_level": "HIGH" if score >= threshold else "LOW",
            "summary": f"Fallback alert: Anomaly score {score} exceeded threshold {threshold}. {subgraph_summary}",
            "key_indicators": [
                {"feature": "score", "value": str(score), "concern": "Exceeded rule threshold"},
                {"feature": "structuring_cluster_count", "value": str(features.get("structuring_cluster_count", 0)),
                 "concern": "Multiple near-threshold transactions"}
            ],
            "recommended_action": "REVIEW",
            "confidence": "MEDIUM",
            "fallback": True,
            "error": str(e)
        }

    return {
        **event,
        "explanation": explanation
    }