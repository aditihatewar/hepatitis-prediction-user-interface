import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('The_Hepatitis_Model.pkl','rb'))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    numeric_features = []
    string_features = []

    # Separate numeric and string features
    for value in request.form.values():
        try:
            numeric_features.append(float(value))
        except ValueError:
            string_features.append(value)

    # Ensure the numeric features are converted to a numpy array
    if numeric_features:
        numeric_features = np.array(numeric_features).reshape(1, -1)
        prediction_numeric = model.predict(numeric_features)
        prediction_text = "The disease is {}".format(prediction_numeric[0])
    else:
        prediction_text = "Unable to make a prediction without numeric features."

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
