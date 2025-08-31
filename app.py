# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from chatbot import render_chatbot, explain_prediction as chatbot_explain
import os

app = Flask(__name__)

# Global variables
model = None
feature_names = None
feature_importances = None
encoders = None
latest_prediction = None
latest_input_values = None
log_file = "prediction_log.csv"

# Load model and encoders once when the app starts
def load_assets():
    global model, feature_names, feature_importances, encoders
    try:
        with open("customer_churn_model.pkl", "rb") as f:
            model_data = pickle.load(f)
            model = model_data["model"]
            feature_names = model_data["feature_names"]
            feature_importances = model_data["feature_importances"]

        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading assets: {e}. Please ensure model and encoder files are in the directory.")
        exit()

load_assets()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global latest_prediction, latest_input_values
    try:
        input_values = {feature: request.form.get(feature) for feature in feature_names}
        input_df = pd.DataFrame([input_values])

        # Convert numeric values
        for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Apply encoders
        for col in input_df.columns:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])

        # Reorder columns to match training
        input_df = input_df[feature_names]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        prediction_proba = proba[prediction]

        # Log prediction to CSV
        log_data = input_values.copy()
        log_data["Prediction"] = "Churn" if prediction == 1 else "No Churn"
        log_data["Probability (%)"] = round(prediction_proba * 100, 2)
        log_df = pd.DataFrame([log_data])

        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)

        latest_prediction = prediction
        latest_input_values = input_values
        
        result_text = "Customer is likely to churn" if prediction == 1 else "Customer is not likely to churn"
        probability_text = f"{'Churn' if prediction == 1 else 'No Churn'} Probability: {round(prediction_proba * 100, 2)}%"

        # Redirect to home with prediction results as query parameters
        return redirect(url_for('home', 
                                prediction_text=result_text, 
                                probability_text=probability_text))
        
    except Exception as e:
        return f"Error during prediction: {e}"

@app.route("/dashboard-data")
def dashboard_data():
    # Default chart in case CSV doesn't exist
    default_chart = {"No Churn": 1, "Churn": 0}

    if not os.path.exists("prediction_log.csv"):
        return jsonify(default_chart)
    
    try:
        df = pd.read_csv("prediction_log.csv")

        # Ensure "Prediction" column exists
        if "Prediction" not in df.columns:
            return jsonify(default_chart)

        # Normalize values: strip spaces, standardize capitalization
        df["Prediction"] = df["Prediction"].astype(str).str.strip().str.title()

        # Count occurrences
        counts = df["Prediction"].value_counts().to_dict()

        # Ensure both categories are present
        chart_data = {
            "No Churn": int(counts.get("No Churn", 0)),
            "Churn": int(counts.get("Churn", 0))
        }

        return jsonify(chart_data)

    except Exception as e:
        print(f"Error reading CSV for dashboard: {e}")
        return jsonify(default_chart)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_message = request.json.get("message")

    if "why" in user_message.lower() or "reason" in user_message.lower():
        if latest_prediction is not None and latest_input_values is not None:
            explanation = chatbot_explain(latest_prediction, latest_input_values, feature_importances)
            return jsonify({"response": explanation})
        else:
            return jsonify({"response": "Please make a prediction first."})
    else:
        # Assuming chatbot.py has a render_chatbot function
        # This part of the code needs to be adjusted based on the actual chatbot.py content
        response = "Sorry, I'm still learning and can only answer questions about the prediction after you submit the form."
        return jsonify({"response": response})

def explain_prediction(prediction, input_features, importances):
    # This function should probably be in chatbot.py
    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:3]

    status = "likely to churn" if prediction else "not likely to churn"
    explanation = f"The customer is {status} based on the following:\n"

    for feature, importance in top_features:
        val = input_features.get(feature, "unknown")
        explanation += f"- {feature} = {val} (importance: {importance:.2f})\n"

    return explanation

if __name__ == "__main__":
    app.run(debug=True)