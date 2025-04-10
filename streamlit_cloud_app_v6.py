import streamlit as st
import requests

# Streamlit UI
st.title("Welcome to the ")
st.image("logo.png")
st.title("Credit Scoring App!")

# Select client credit application reference
selected_value = st.select_slider(
    "Select a client credit application reference:",
    options=range(1, 46128)
)

# Display the selected value
st.write(f"You selected client application: {selected_value}")

# IMPORTANT: In advanced settings, choose Python 3.10 when deploying the app in Streamlit Cloud
# to avoid errors related to distutils (discontinued in Python 3.12 onwards).

# Send a POST request to the API using the selected value
response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}")

# Display the response from the API (optional)
if response.status_code == 200:
    st.write(f"API Response: {response.text}")
else:
    st.error(f"Failed to fetch data. Status code: {response.status_code}")

# Main function placeholder (optional)
def main():
    pass

if __name__ == "__main__":
    main()
