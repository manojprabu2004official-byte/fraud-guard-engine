# tests/load/benchmark_tier1.py
import time
import requests
import numpy as np

API_URL = "http://localhost:8000/v1/screen"
NUM_REQUESTS = 1000

payload_sample = {
    "transaction_id": "BENCH_001",
    "sender_id": "Acc_9901",
    "receiver_id": "Acc_1102",
    "amount": 25000.0,
    "oldbalanceOrg": 25000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "is_transfer": 1
}

def run_benchmark():
    latencies = []
    print(f"Executing {NUM_REQUESTS} sequential requests against Tier-1 Service...")
    
    session = requests.Session()
    
    for i in range(NUM_REQUESTS):
        start = time.perf_counter()
        resp = session.post(API_URL, json=payload_sample)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if resp.status_code == 200:
            latencies.append(elapsed_ms)
        else:
            print(f"Error: Received status {resp.status_code}")
            
    latencies = np.array(latencies)
    
    print("\n================ TIER-1 LATENCY BENCHMARK ================")
    print(f"Total Requests Processed: {len(latencies):,}")
    print(f"Mean Latency:             {np.mean(latencies):.2f} ms")
    print(f"p50 (Median) Latency:     {np.percentile(latencies, 50):.2f} ms")
    print(f"p95 Latency:              {np.percentile(latencies, 95):.2f} ms")
    print(f"p99 Latency (Target <15): {np.percentile(latencies, 99):.2f} ms")
    print(f"Max Single Latency:       {np.max(latencies):.2f} ms")
    print("==========================================================")

if __name__ == "__main__":
    run_benchmark()