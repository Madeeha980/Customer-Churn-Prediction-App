# Customer-Churn-Prediction-App
Project Overview

This project predicts customer churn for a telecom company. It uses a Random Forest Classifier (best-performing model) trained on customer data. Users can input customer details through a web interface built with Flask, get predictions, and view a dashboard showing churn statistics. A chatbot explains the predictions based on feature importance.

Features

Predict customer churn based on input data.

Interactive dashboard showing Churn vs No Churn statistics.

Chatbot explains the prediction and highlights key factors influencing the decision.

Model and preprocessing pipelines are saved for reuse.

Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-learn, XGBoost, SMOTE

Frontend: HTML, CSS, JavaScript (with Chart.js for charts)

Data Handling: Pandas, NumPy

Visualization: Seaborn, Matplotlib

Other Libraries: Pickle (for saving models and encoders)

Installation Instructions

Clone the repository or download manually:

git clone <your-repo-url>
cd customer-churn-app


Install required Python packages:

pip install -r requirements.txt


If requirements.txt is not available, install manually:

pip install flask pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn


Run the Flask app:

python app.py


Open your browser and navigate to:

http://127.0.0.1:5000/

Project Structure
customer-churn-app/
│
├── app.py                   # Flask application
├── customer_churn_model.pkl # Saved ML model
├── encoders.pkl             # Saved LabelEncoders
├── sencoders.pkl            # Saved StandardScalers
├── notebook.ipynb           # Jupyter notebook with data preprocessing and model training
├── prediction_log.csv       # Logs user predictions (generated automatically)
└── templates/
    └── index.html           # Frontend HTML page

How to Use

Enter customer details in the form (tenure, services, charges, etc.).

Click Predict Churn to see the result.

View the churn summary chart in the right panel.

Ask the chatbot why the prediction was made to get a brief explanation.

Model & Preprocessing

Feature Encoding: Categorical variables are label-encoded using encoders.pkl.

Scaling: Numerical features (tenure, MonthlyCharges, TotalCharges) are standardized using sencoders.pkl.

Model: Random Forest Classifier trained with SMOTE for handling class imbalance.

Saved Objects: The trained model, encoders, and scalers are saved as .pkl files for reuse in the app.

Future Improvements

Add more visualization options (bar charts, trend analysis).

Improve the chatbot with NLP for free-text queries.

Add user authentication for saving predictions.

Deploy the app online using Heroku or AWS.

License

This project is open-source and free to use for educational purposes.
