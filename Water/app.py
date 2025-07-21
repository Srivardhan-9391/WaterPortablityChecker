from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('water_potability_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract input data
        ph_value = float(data['ph_value'])
        Hardness = float(data['Hardness'])
        Solids = float(data['Solids'])
        Chloramines = float(data['Chloramines'])
        Sulfate = float(data['Sulfate'])

        # Combine inputs into a single array
        input_features = np.array([[ph_value, Hardness, Solids, Chloramines, Sulfate]])
        
        # Scale the input features
        input_features_scaled = scaler.transform(input_features)

        # Predict potability
        prediction = model.predict(input_features_scaled)[0]
        potability = "Potable" if prediction == 1 else "Not potable"

        # Return the prediction
        return jsonify({"potability": potability})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
