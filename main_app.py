from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load Pipeline
model = joblib.load("pepperfry_pipeline.pkl")


@app.route("/")
def home():
    return "Pepperfry Price Prediction API"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Create DataFrame
    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)

    return jsonify({
        "Predicted Price": round(float(prediction[0]), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)