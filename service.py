
import bentoml
import numpy as np
import pandas as pd


from bentoml.models import BentoModel

# Define the runtime environment for the Bento
demo_image = bentoml.images.PythonImage(python_version="3.13.2").python_packages("mlflow", "lightgbm")

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class creditclassifier:
    # Declare the model as a class attribute
    bento_model = BentoModel("credit_scoring:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    # Define an API endpoint
    @bentoml.api
    def predict(self, input_data:pd.DataFrame):
        preds = self.model.predict(input_data)
        return preds.tolist()
    
