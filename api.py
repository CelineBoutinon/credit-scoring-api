import flask
from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)
app.config["DEBUG"] = True

# Load client test data
data_path = "..\..\PROJET 7\X_test_final.csv"
client_data=pd.read_csv(data_path)


# Get cwd
current_directory = os.path.dirname(os.path.abspath(__file__))

# Charge model outside of if __name__ == "__main__" clause:
model_path = os.path.join(current_directory, "..", "Simulations", "Best_model", "model.pkl")
model = joblib.load() # load champion

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Credit Scoring</h1>
<p>Welcome to the HOME CREDIT Credit Scoring app.</p>
<p>Use /client?id=<ID> to access client demographics.</p>
<p>Use predict/client?id=<ID to retrieve client credit application decision.</p>'''


@app.route('/client', methods=['GET'])
def api_id():
    # Check client basic demographics
    if 'id' in request.args:
        try:
            # Convert ID to integer
            id = int(request.args['id'])
        except ValueError:
            return "Error: Client id must contain only digits (max. 5).", 400
    else:
        return "Error: Please provide a client id.", 400
    
        # Ensure client id exists in test data
    if id < 0 or id >= len(client_data):            
        return "Error: Client id not in client database.", 404

    results = []
 
    results.append(client_data.iloc[[id]].to_json(orient='records')) # redundant with jsonify below?
 
    return jsonify(results) #http://127.0.0.1:5000/client?id=42 in browser returns info for client at index 42 in test data

    # modify return to include only demographis such as income type, employment sector, year birth as age, M/F, white collar, upper ed, married

 
@app.route("/predict/<id>", methods=['POST']) # or ['GET'] ??
def predict():
    data = request.json

    # e=Get xclient id from url
    id = request.args.get('id_client', default=42, type=int)
    
    # Build relative path to client test data
    csv_path = os.path.join(current_directory, "..", "Simulations", "Data", "df_train.csv")

    # Load test data
    client_data = pd.read_csv(csv_path)
    client_particulars = client_data.iloc[[id]] # .values ?

    # Consider dropping df index for prediction
    
    # Prédire
    prediction = model.predict_proba(client_particulars) #.predict not predict_proba?
    proba = prediction[0][1] # prediction[0][0] is proba of client NOT defaulting
    if proba > 0.502:
        proba_class = 1
        decision = "Reject loan application."
    else:
        proba_class = 0
        decision = "Grant loan."

    # shap won't work with MLFlow pyfunc model
    
    # Retourner les valeurs SHAP avec la probabilité
    return jsonify({
        'client id': id,
        'client default probability': proba, 
        'class': proba_class,
        'decision': decision
    })








if __name__ == "__main__":
    # port = os.environ.get("PORT", 5000)
    # app.run(debug=False, host="0.0.0.0", port=int(port))
    app.run()