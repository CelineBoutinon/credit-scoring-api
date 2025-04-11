import streamlit as st
import requests
import json
import shap
import matplotlib.pyplot as plt

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
app_response = requests.get(f"https://credit-scoring-api-0p1u.onrender.com/predict/{selected_value}")
app_data = json.loads(app_response)


# Display the response from the API (optional)
if app_response.status_code == 200:
    # st.write(f"API Response: {response.text}")

    # st.write(f"Client Application No.: {data['Client id']}")
    # st.write(f"Client default probability: {data['Client default probability']}")
    # st.write(f"Class: {data['Class']}")
    # st.write(f"Decision: {data['Decision']}")
    
    st.write(app_data)

else:
    st.error(f"Failed to fetch data. Status code: {app_response.status_code}")


# if shap_values is not None:
#                 # Create SHAP waterfall plot for the first prediction
#                 fig, ax = plt.subplots(figsize=(10,6))
#                 st.title(f"Key decision factors for client {index} :")
#                 shap.plots.waterfall(shap.Explanation(values=shap_values[index],
#                                                             base_values=explainer.expected_value,
#                                                             data=X_test.loc[index],
#                                                             feature_names=X_test.columns),
#                                                             show=True,
#                                                             max_display=6)
#                 st.pyplot(fig)






# Main function placeholder (optional)
def main():
    pass

if __name__ == "__main__":
    main()
