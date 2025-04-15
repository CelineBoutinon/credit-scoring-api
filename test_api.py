# run from the command line with py -m pytest test_api.py -v -s
# or py -m pytest test_api.py -v -s -W ignore::Warning to ignore the sklearn tags warning

# import os
# import sys
import joblib
import pandas as pd
# from flask import Flask, jsonify, request
import app
# import pytest
import requests
import pandas as pd
from flask import json
from app import app

# Base URL for the API
base_url = "http://127.0.0.1:5000"

# Test that the prediction endpoint returns a successful response
def test_prediction():
    response = requests.get(f"{base_url}/predict/1")
    assert response.status_code == 200

# Test that the client demographics endpoint returns a successful response
def test_client_demographics():
    response = requests.get(f"{base_url}/client/1")
    assert response.status_code == 200

# Test that the client demographics endpoint returns an error for invalid client ID
def test_client_demographics_invalid_id():
    response = requests.get(f"{base_url}/client/46129")
    assert response.status_code == 404

# Test that the prediction endpoint returns an error for invalid client ID
def test_prediction_invalid_id():
    response = requests.get(f"{base_url}/predict/46129")
    assert response.status_code == 500

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

# Test that the prediction returns a default probability
def test_prediction():
    # Load test data
    df = pd.read_csv('X_test_final.csv')

    # Get the index from the first row of your test data
    index_value = df.index[0]  # Get the first index value

    with app.test_client() as client:
        # Make a request to your prediction endpoint, matching your API route
        response = client.get(f'/predict/{index_value}')  # Use .get() and f-string

        # Parse the JSON response
        data = json.loads(response.data)

        # Extract the prediction (adjust key to match your API response)
        prediction = data['Client default probability']  # Assuming your API returns {'prediction': value}

        # Assert that the prediction is not None
        assert prediction is not None, "Failed to predict application outcome for client."

        # Optionally, add more specific assertions about the prediction value
        assert isinstance(prediction, (int, float)), "Prediction should be a number."


