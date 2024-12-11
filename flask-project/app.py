from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mock_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Model file not found. Continuing without the model.")
    model = None

# App part

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/tryModel')
def subpage1():
    return render_template('tryModel.html')

@app.route('/about')
def subpage2():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please make sure mock_model.joblib exists.'}), 500

    try:
        data = request.json
        if not data or 'inputs' not in data or len(data['inputs']) != 4:
            return jsonify({'error': 'Invalid input. Please provide an array of 4 numbers.'}), 400

        inputs = np.array(data['inputs']).reshape(1, -1)

        prediction = model.predict(inputs)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    #app.run(debug=True)
