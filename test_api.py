# run from the command line with py -m pytest test_api.py -v -s

import joblib
import pandas as pd
import app
import pytest
import requests
import random
import pandas as pd
from flask import json
from app import app

# Base URL for the API
# base_url = "http://127.0.0.1:5000" # for local tests
base_url = "https://credit-scoring-api-0p1u.onrender.com/" # for cloud app tests

# Test that the prediction endpoint returns a successful response
def test_prediction():
    random_client_id = random.randint(1, 46128)
    response = requests.get(f"{base_url}/predict/{random_client_id}")
    assert response.status_code == 200

# Test that the client demographics endpoint returns a successful response
def test_client_demographics():
    random_client_id = random.randint(1, 46128)
    response = requests.get(f"{base_url}/client/{random_client_id}")
    assert response.status_code == 200

# Test that the client demographics endpoint returns an error for an invalid client ID
def test_client_demographics_invalid_id():
    response = requests.get(f"{base_url}/client/46129")
    assert response.status_code == 404

# Test that the prediction endpoint returns an error for an invalid client ID
def test_prediction_invalid_id():
    response = requests.get(f"{base_url}/predict/46129")
    assert response.status_code == 500

# Test that the prediction endpoint returns the expected JSON structure
def test_prediction_json_structure():
    random_client_id = random.randint(1, 46128)
    response = requests.get(f"{base_url}/predict/{random_client_id}")
    assert response.status_code == 200
    data = response.json()
    expected_keys = ['Client id', 'Client default probability', 'Class', 'Decision', 'Key Decision Factors',
                     'Expected Shap Value', 'Shap values client']
    assert all(key in data for key in expected_keys)

# Test that the champion MLFlow model loads
def test_model_loading():
    model = joblib.load('final_model.joblib')
    assert model is not None, "Error loading model."

# Test that the client test data csv file loads
def test_csv_loading():
    df = pd.read_csv('X_test_final.csv')
    assert not df.empty, "Error loading csv file."

# Test that the prediction returns a default probability and that it is a number
def test_prediction():
    df = pd.read_csv('X_test_final.csv')
    random_client_id = random.randint(1, 46128)
    index_value = df.index[random_client_id]
    with app.test_client() as client:
        response = client.get(f'/predict/{index_value}')
        data = json.loads(response.data)
        prediction = data['Client default probability']
        assert prediction is not None, "Failed to predict application outcome for client."
        assert isinstance(prediction, (int, float)), "Prediction should be a number."