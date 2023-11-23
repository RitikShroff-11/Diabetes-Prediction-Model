import numpy as np
from flask import Flask, request, render_template
import pickle
# from sklearn.linear_model import LogisticRegression  # Import your model here

# # Initialize and train your model
# regression_model = LogisticRegression()  # Initialize your model here
# # Assume you have already trained your model with X_train, y_train

model=pickle.load(open('model.pkl',"rb"))

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        val1 = float(request.form['pregnancies'])
        val2 = float(request.form['glucose'])
        val3 = float(request.form['bloodPressure'])
        val4 = float(request.form['skinThickness'])
        val5 = float(request.form['insulin'])
        val6 = float(request.form['bmi'])
        val7 = float(request.form['diabetesPedigree'])
        val8 = float(request.form['age'])  # Convert 'age' to float

        arr = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
        pred = model.predict(arr)

        return render_template("Result.html", result=pred )

if __name__ == "__main__":
    app.run(debug=True)


