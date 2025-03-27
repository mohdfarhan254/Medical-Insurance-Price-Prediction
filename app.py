# ------------------- Flask App for Medical Insurance Price Prediction -------------------
# This app has both Web Frontend + API endpoint

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained ML model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# ------------------- Web Route -------------------
@app.route('/')
def home():
    # Render HTML page with input fields
    return render_template('index.html')

# ------------------- Predict from Web Form -------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])

        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=f'Estimated Price: ₹{prediction}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

# ------------------- Predict from API (Postman/React) -------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json
        age = float(data['age'])
        sex = int(data['sex'])
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = int(data['smoker'])
        region = int(data['region'])

        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return jsonify({'prediction': f'₹{prediction}'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
