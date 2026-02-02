"""Test the FastAPI service locally.

Usage:
  1. Start the server:
     cd transformer_co2 && uvicorn deployment.app:app --reload --port 8000

  2. Run this script (in another terminal):
     cd transformer_co2 && python deployment/test_api.py

  Or use curl directly (see examples at bottom).
"""

import json
import sys
import os

# --- Option A: Test via HTTP (requires running server) ---
def test_via_http():
    import requests

    BASE = "http://localhost:8000"

    # 1. Health check
    print("=== Health Check ===")
    r = requests.get(f"{BASE}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Body:   {r.json()}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("  PASS\n")

    # 2. Predict with real data
    print("=== Predict (real test data) ===")
    # Load a real sample from the data pipeline
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_pipeline import load_excel_files

    df_list, paths = load_excel_files()
    test_df = df_list[2]  # 140207_1.xlsx
    test_df = test_df.set_index("time")

    # Extract 19 timesteps of raw features + labels
    feature_cols = sorted([c for c in test_df.columns if c != "label"])
    readings = []
    for i in range(19):
        row = test_df.iloc[i]
        readings.append({
            "features": row[feature_cols].tolist(),
            "sampling_point": int(row["label"]),
        })

    payload = {"readings": readings, "experiment_id": "test_140207_1"}
    r = requests.post(f"{BASE}/predict", json=payload)
    print(f"  Status: {r.status_code}")
    resp = r.json()
    print(f"  Model:  {resp['model_version']}")
    print(f"  Predictions (step 0):")
    for pt in resp["predictions"][0]:
        print(f"    Point {pt['point']}: {pt['co2_percent']:.4f} CO2 %")
    assert r.status_code == 200
    assert len(resp["predictions"]) > 0
    print("  PASS\n")

    # 3. Predict with too few readings (should fail)
    print("=== Predict (too few readings) ===")
    short_payload = {"readings": readings[:5]}
    r = requests.post(f"{BASE}/predict", json=short_payload)
    print(f"  Status: {r.status_code} (expected 400)")
    assert r.status_code == 400
    print("  PASS\n")

    print("All HTTP tests passed!")


# --- Option B: Test without server (direct FastAPI TestClient) ---
def test_via_testclient():
    from fastapi.testclient import TestClient

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deployment.app import app

    with TestClient(app) as client:

        # 1. Health
        print("=== Health Check ===")
        r = client.get("/health")
        print(f"  {r.json()}")
        assert r.status_code == 200
        print("  PASS\n")

        # 2. Predict with dummy data
        print("=== Predict (dummy data) ===")
        readings = [{"features": [0.5] * 90, "sampling_point": (i % 6) + 1} for i in range(19)]
        r = client.post("/predict", json={"readings": readings})
        print(f"  Status: {r.status_code}")
        resp = r.json()
        for pt in resp["predictions"][0]:
            print(f"    Point {pt['point']}: {pt['co2_percent']:.4f} CO2 %")
        assert r.status_code == 200
        print("  PASS\n")

        # 3. Predict with real data
        print("=== Predict (real data) ===")
        from src.data_pipeline import load_excel_files
        df_list, paths = load_excel_files()
        test_df = df_list[2].set_index("time")
        feature_cols = sorted([c for c in test_df.columns if c != "label"])
        readings = []
        for i in range(19):
            row = test_df.iloc[i]
            readings.append({
                "features": row[feature_cols].tolist(),
                "sampling_point": int(row["label"]),
            })
        r = client.post("/predict", json={"readings": readings})
        resp = r.json()
        print(f"  Status: {r.status_code}")
        for pt in resp["predictions"][0]:
            print(f"    Point {pt['point']}: {pt['co2_percent']:.4f} CO2 %")
        assert r.status_code == 200
        print("  PASS\n")

        print("All TestClient tests passed!")


if __name__ == "__main__":
    if "--http" in sys.argv:
        test_via_http()
    else:
        print("Running with FastAPI TestClient (no server needed).\n"
              "Use --http to test against a running server.\n")
        test_via_testclient()
