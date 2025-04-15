import pytest
import streamlit as st
import requests
import json
from unittest.mock import patch, Mock
import responses

# Mock API response
def mock_api_response():
    return {
        "Class": "default",
        "Client default probability": 0.6659363614533874,
        "Client id": 1,
        "Decision": "reject loan application",
        "Expected Shap Value": -0.17939734770419566,
        "Key Decision Factors": [["num__TOTAL_ACTIVE_CONSUMER_LOANS", 0.1781375309499465]],
        "Shap values client": '[{"num__AMT_CREDIT":0.0280267594}]'
    }

# Test that the app displays the client default probability correctly
@responses.activate
def test_client_default_probability(capsys):
    responses.add(responses.GET, "https://credit-scoring-api-0p1u.onrender.com/predict/1",
                  json=mock_api_response(), status=200)

    # Run the app with a mock API response
    import streamlit_cloud_app_v6  # Import your Streamlit app script
    streamlit_cloud_app_v6.main()  # Assuming your app has a main function

    # Capture the output
    captured = capsys.readouterr()
    assert "Client default probability: 66.59%" in captured.out

# Test that the app handles API errors correctly
@responses.activate
def test_api_error_handling(capsys):
    responses.add(responses.GET, "https://credit-scoring-api-0p1u.onrender.com/predict/1",
                  status=404)

    # Run the app with a mock API error
    import streamlit_cloud_app_v6  # Import your Streamlit app script
    streamlit_cloud_app_v6.main()  # Assuming your app has a main function

    # Capture the output
    captured = capsys.readouterr()
    assert "Failed to fetch data. Status code : 404" in captured.out
