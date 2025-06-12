import os
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)  # ✅ fixed __name__

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Crop Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Expect JSON body from Android app
        data = request.get_json()

        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # ✅ Return error if something goes wrong

if __name__ == "__main__":  # ✅ fixed __main__
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
