# apps/tier1_inference/main.py
import time
import json
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from confluent_kafka import Producer
from pathlib import Path

app = FastAPI(
    title="Tier-1 Fast Screening Engine",
    description="Sub-15ms screening gatekeeper for high-throughput payment streams",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Check parent directory if models is located outside the inner folder
import os

def find_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "models" / "artifacts" / "xgboost.onnx"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "xgboost.onnx not found. Run the training/export script first, "
        "or set the MODEL_PATH environment variable."
    )

MODEL_PATH = find_model_path()
KAFKA_BOOTSTRAP = "localhost:9092"
SUSPICIOUS_TOPIC = "transactions.suspicious"

# Load ONNX Session with optimized execution settings
session = None
input_name = None
kafka_producer = None

@app.on_event("startup")
def load_artifacts():
    global session, input_name, kafka_producer
    
    # 1. Initialize ONNX Runtime Session (C++ Backend)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. Initialize Async Kafka Producer for Tier-2 Escalations
    if os.getenv("ENABLE_KAFKA", "1") == "1":
        producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'queue.buffering.max.messages': 100000,
            'queue.buffering.max.ms': 10,
            'acks': 1
        }
        try:
            kafka_producer = Producer(producer_config)
        except Exception as e:
            print(f"Warning: Kafka Producer connection deferred ({e})")

# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., example="TX_99218401")
    sender_id: str = Field(..., example="Acc_A10092")
    receiver_id: str = Field(..., example="Acc_B88201")
    amount: float = Field(..., example=45000.00)
    oldbalanceOrg: float = Field(..., example=45000.00)
    newbalanceOrig: float = Field(..., example=0.00)
    oldbalanceDest: float = Field(..., example=0.00)
    newbalanceDest: float = Field(..., example=0.00)
    is_transfer: int = Field(..., example=1, description="1 for TRANSFER, 0 for CASH_OUT")

class ScreeningResult(BaseModel):
    transaction_id: str
    fraud_score: float
    action: str
    latency_ms: float

# -----------------------------------------------------------------------------
# Screening Endpoint (Sub-15ms Target)
# -----------------------------------------------------------------------------
@app.post("/v1/screen", response_model=ScreeningResult)
async def screen_transaction(payload: TransactionPayload):
    start_time = time.perf_counter()
    
    # 1. Calculate inline feature discrepancies
    error_balance_orig = payload.oldbalanceOrg - payload.amount - payload.newbalanceOrig
    error_balance_dest = payload.oldbalanceDest + payload.amount - payload.newbalanceDest
    drain_ratio = (payload.amount / payload.oldbalanceOrg) if payload.oldbalanceOrg > 0 else 0.0
    
    # 2. Construct float32 feature array for ONNX
    features = np.array([[
        payload.amount,
        payload.oldbalanceOrg,
        payload.newbalanceOrig,
        payload.oldbalanceDest,
        payload.newbalanceDest,
        error_balance_orig,
        error_balance_dest,
        payload.is_transfer,
        drain_ratio
    ]], dtype=np.float32)
    
    # 3. Execute ONNX C++ Inference
    outputs = session.run(None, {input_name: features})
    
    # Extract fraud probability (Class 1)
    fraud_score = float(outputs[1][0][1])
    
    # 4. Decision Logic & Asynchronous Tier-2 Escalation
    is_suspicious = fraud_score > 0.85
    action = "ESCALATE_TIER2" if is_suspicious else "PASS"
    
    if is_suspicious and kafka_producer is not None:
        escalation_payload = {
            "transaction_id": payload.transaction_id,
            "sender_id": payload.sender_id,
            "receiver_id": payload.receiver_id,
            "amount": payload.amount,
            "fraud_score": fraud_score,
            "timestamp": time.time()
        }
        kafka_producer.produce(
            SUSPICIOUS_TOPIC,
            key=payload.sender_id.encode('utf-8'),
            value=json.dumps(escalation_payload).encode('utf-8')
        )
        kafka_producer.poll(0)  # Non-blocking flush
    
    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
    
    return ScreeningResult(
        transaction_id=payload.transaction_id,
        fraud_score=round(fraud_score, 4),
        action=action,
        latency_ms=latency_ms
    )