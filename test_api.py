import pytest
import requests
import json

# Base URL for the API
base_url = "http://127.0.0.1:5000"

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
