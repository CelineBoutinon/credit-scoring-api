import os
import sys
import joblib
import pandas as pd
import pytest
from flask import Flask, jsonify, requests

# Base URL for the API
base_url = "http://127.0.0.1:5000"

# @pytest.fixture
# def client():
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         yield client

# Test that the home page returns a successful response
def test_home():
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200

# Test that the client demographics endpoint returns a successful response
def test_client_demographics():
    response = requests.get(f"{base_url}/client/1")
    assert response.status_code == 200

# Test that the client demographics endpoint returns an error for invalid client ID
def test_client_demographics_invalid_id():
    response = requests.get(f"{base_url}/client/46129")
    assert response.status_code == 404

# Test that the prediction endpoint returns a successful response
def test_prediction():
    response = requests.get(f"{base_url}/predict/1")
    assert response.status_code == 200

# Test that the prediction endpoint returns an error for invalid client ID
def test_prediction_invalid_id():
    response = requests.get(f"{base_url}/predict/46129")
    assert response.status_code == 404

# Test that the prediction endpoint returns the expected JSON structure
def test_prediction_json_structure():
    response = requests.get(f"{base_url}/predict/1")
    assert response.status_code == 200
    data = response.json()
    expected_keys = ['Client id', 'Client default probability', 'Class', 'Decision', 'Key Decision Factors', 'Expected Shap Value', 'Shap values client']
    assert all(key in data for key in expected_keys)


def test_model_loading():
    model = joblib.load('final_model.joblib')
    assert model is not None, "Error loading model."

def test_csv_loading():
    df = pd.read_csv('X_test_final.csv')
    assert not df.empty, "Error loading csv file."

def test_prediction():
    import os
    import pandas as pd
    from flask import json
    df = pd.read_csv('X_test_final.csv')
    sk_id_curr = df.iloc[0]['SK_ID_CURR']
    with app.test_client() as client:
        response = client.post('/predict', json={'SK_ID_CURR': sk_id_curr})
        data = json.loads(response.data)
        prediction = data['probability']
        assert prediction is not None, "Failed to predict application outcome for client."

