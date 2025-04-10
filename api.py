# launch app by running flask --app api.py run --debug from the command line

# import flask
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import pandas as pd
# import mlflow
# import mlflow.pyfunc

app = Flask(__name__)
app.config["DEBUG"] = True

# Load client test data
# data_path = "..\..\PROJET 7\X_test_final.csv"
client_data=pd.read_csv('X_test_final.csv')

# model_uri = <TBD>
# model = mlflow.pyfunc.load_model(model_uri)


# Get cwd
# current_directory = os.path.dirname(os.path.abspath(__file__))

# Charge model outside of if __name__ == "__main__" clause:
# model_path = os.path.join(current_directory, "..", "Simulations", "Best_model", "model.pkl")
model = joblib.load('final_model.joblib') # load champion

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Credit Scoring</h1>
<p>Welcome to the HOME CREDIT Credit Scoring app.</p>
<p>- use /client/ID to access client demographics</p>
<p>- use /predict/ID to retrieve client credit application decision</p>
<p>where ID is the client's unique Home Credit application number (whole number between 1 and 46128)</p>
'''


@app.route('/client/<int:id>', methods=['GET'])
def api_id(id):
    # Display summary client demographics

    # if 'id' in request.args:
    #     try:
            # Convert ID to integer
    #         id = int(request.args['id']) # or request.args.get('id') ?
    #     except ValueError:
    #         return "Error: Client id must contain only digits (max. 5).", 400
    # else:
    #     return "Error: Please provide a client id.", 400
    
        # Ensure client id exists in test data
    if (id-1) >= client_data.shape[0]:            
        return "Error: Client id not in application database. Enter a whole number between 1 and 46128.", 404
    if ((id-1) < 0):
        return "Error: Client id not in application database. Enter a whole number between 1 and 46128.", 404

    result_cols = ['INCOME_TYPE', 'EMPLOYMENT_SECTOR', 'DISPOSABLE_INCOME_per_capita', 'YEAR_BIRTH', 'CREDIT_RATING', 'CLIENT_BAD_CREDIT_HISTORY', 'CLIENT_FRAUD_FLAG',
                   'IS_MALE', 'WHITE_COLLAR', 'UPPER_EDUCATION', 'IS_MARRIED', 'LIVES_INDEPENDENTLY']
    results = []

    row_data = client_data.loc[id-1, result_cols].to_dict()
    for k, v in row_data.items():
        if v==0:
            row_data[k] = 'no'
        if v==1:
            row_data[k] = 'yes'
        
    row_data['AGE'] = row_data.pop('YEAR_BIRTH')
    
    results.append(row_data)
  
    return jsonify('Client Home Credit application number:', id, 'Client summary information:', results)
    #http://127.0.0.1:5000/client/42 in browser returns info for client at index 42 in test data

  
 
@app.route("/predict/<int:id>", methods=['GET'])
def predict(id): # Get `id` directly from the URL
    # Load client data
    client_particulars = client_data.iloc[[int(id-1)]] 
    client_particulars = client_data.iloc[[id-1]] # .values ? is request needed at all?

    # data = request.json

    # Get xclient id from url
    # id = request.args.get('id_client', default=42, type=int)

    # Example URL: /predict/42 or /predict?id_client=42 - USE EITHER
    # id_from_url = request.view_args.get('id')  # From `<id>` in URL
    # id_from_query = request.args.get('id_client')  # From query string
    
    # Build relative path to client test data
    # csv_path = os.path.join(current_directory, "..", "xx", "xx", "X_test_final.csv")

    # Load test data
    # client_data = pd.read_csv(csv_path)
    # client_particulars = client_data.iloc[[id]] # .values ?

    
    # Predict outcome of client credit application
    prediction = model.predict_proba(client_particulars)
    proba = prediction[0][1] # prediction[0][0] is proba of client NOT defaulting
    if proba > 0.502:
        proba_class = 'default'
        decision = "reject loan application"
    else:
        proba_class = 'no default'
        decision = "grant loan"

    # shap won't work with MLFlow pyfunc model
    
    # Return application decision
    return jsonify({
        'Client id': id,
        'Client default probability': proba, 
        'Class': proba_class,
        'Decision': decision
    })


if __name__ == "__main__":
    # port = os.environ.get("PORT", 5000)
    # app.run(debug=False, host="0.0.0.0", port=int(port))
    app.run()