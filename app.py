# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ==========================
# Global variables
# ==========================
model = None
feature_names = None
feature_importances = None
encoders = None
sencoders = None
latest_prediction = None
latest_input_values = None
log_file = "prediction_log.csv"

# ==========================
# Load model + encoders + scalers
# ==========================
def load_assets():
    global model, feature_names, feature_importances, encoders
    try:
        # Load model
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
            feature_names = model_data["feature_names"]
            feature_importances = model_data["feature_importances"]

        # Load encoders
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

    except FileNotFoundError as e:
        print(f"Error loading assets: {e}")
        exit()



load_assets()

# ==========================
# Routes
# ==========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global latest_prediction, latest_input_values

    try:
        # Collect form values
        input_values = {feature: request.form.get(feature) for feature in feature_names}
        input_df = pd.DataFrame([input_values])

        # Convert numeric safely
        for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # 🔹 Replace "No internet service" with "No"
        replace_cols = [
            'InternetService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in replace_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace("No internet service", "No")

        # 🔹 Apply label encoders
        for col in input_df.columns:
            if col in encoders:
                try:
                    input_df[col] = encoders[col].transform(input_df[col])
                except ValueError:
                    fallback = "No" if "No" in encoders[col].classes_ else encoders[col].classes_[0]
                    input_df[col] = encoders[col].transform([fallback])

        # Reorder columns exactly as in training
        input_df = input_df[feature_names]

        # ✅ Debug print — check values going to model
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)   # so it doesn't wrap lines

        print("\n====== DEBUG: Input DF before prediction ======")
        print(input_df)
        print("==== Final Encoded Input (Flask) ====")
        
        print("Dtypes:")
        print(input_df.dtypes)



        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]  # [No Churn, Churn]

        no_churn_prob = round(proba[0] * 100, 2)
        churn_prob = round(proba[1] * 100, 2)

        # Save probabilities for pie chart
        latest_prediction = {"No Churn": no_churn_prob, "Churn": churn_prob}
        latest_input_values = input_values

        # Log prediction
        log_data = input_values.copy()
        log_data["Prediction"] = "Churn" if prediction == 1 else "No Churn"
        log_data["Churn Probability (%)"] = churn_prob
        log_data["No Churn Probability (%)"] = no_churn_prob

        log_df = pd.DataFrame([log_data])
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

        # Output text
        result_text = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"
        probability_text = f"No Churn: {no_churn_prob}% | Churn: {churn_prob}%"

        return redirect(url_for("home",
                                prediction_text=result_text,
                                probability_text=probability_text))

    except Exception as e:
        return f"Error during prediction: {e}"


@app.route("/dashboard-data")
def dashboard_data():
    """Return latest probabilities for pie chart"""
    global latest_prediction
    if latest_prediction is None:
        return jsonify({"No Churn": 0, "Churn": 0})
    return jsonify(latest_prediction)


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
