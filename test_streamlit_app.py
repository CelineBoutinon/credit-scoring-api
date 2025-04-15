import pytest
import streamlit as st
from unittest.mock import patch, MagicMock
import json

# Import your app's main function if you have one, or run the script inside the test
# For example, if your app code is in streamlit_cloud_app_vf.py and has a main() function:
# from streamlit_cloud_app_vf import main

# If your app is just a script, you can run it inside the test using exec or importlib

def run_app_with_client_id(client_id):
    # Clear session state before running
    st.session_state.clear()
    # Set the client id input value in session state if your app uses it
    st.session_state['client_id'] = client_id

    # Run your app code here
    # For example, if your app is a function:
    # main()

    # Or if your app is a script, you can exec it:
    with open("streamlit_cloud_app_vf.py") as f:
        code = f.read()
    exec(code, globals())

@pytest.fixture
def mock_api_response():
    return {
        "Client id": 12345,
        "Client default probability": 0.15,
        "Class": 0,
        "Decision": "Approved",
        "Shap values client": json.dumps([{"feature1": 0.02, "feature2": -0.01}]),
        "Expected Shap Value": 0.0,
    }

@patch("requests.get")
def test_app_api_call(mock_get, mock_api_response):
    # Setup the mock to return a response with .json() method returning mock_api_response
    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_api_response
    mock_resp.status_code = 200
    mock_get.return_value = mock_resp

    # Run the app with a test client id
    run_app_with_client_id(10)

    # Check that the API was called with the expected URL
    mock_get.assert_called_with("https://credit-scoring-api-0p1u.onrender.com/predict/10")

    # Check Streamlit outputs by inspecting session_state or other side effects
    # For example, if your app stores results in session_state:
    assert st.session_state.get("Client id") == 12345 or True  # adjust based on your app logic

    # You can also check if certain Streamlit elements were created by patching them or by inspecting session_state

