import numpy as np
from flask import Flask, request, render_template

import pickle

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/home")
def home():
    return render_template("HomePage PredictionModel1.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Use request.form instead of request.form.values()
    float_features = [float(request.form[x]) for x in request.form]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    # Fix the string formatting in render_template
    return render_template("form PredictionModel1.html", prediction_text=f"Output: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
