from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pepperfry_pipeline.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "message": "Pepperfry Price Prediction API is Running 🚀"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)

        return jsonify({
            "Predicted Price": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500