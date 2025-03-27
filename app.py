from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # To allow requests from frontend (React, etc.)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Medical Insurance Price Prediction Model API"

# API route for JSON POST request
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json  # Get JSON data
        age = float(data['age'])
        sex = int(data['sex'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = int(data['smoker'])
        region = int(data['region'])

        # Convert input into a 2D array
        input_data = np.array([[age, sex, bmi, children, smoker, region]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return jsonify({'prediction': f'₹{prediction}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
