# chatbot.py
def explain_prediction(probability, input_values, feature_names, feature_importances):
    sorted_features = sorted(
        zip(feature_names, feature_importances, input_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_features = sorted_features[:3]

    explanation = f"The predicted churn probability is {probability:.2f}. "
    explanation += "This is mainly influenced by:\n"
    for feature, importance, value in top_features:
        explanation += f"- {feature}: {value} (importance: {importance:.2f})\n"
    return explanation


def render_chatbot(user_input, prediction=None, probability=None, feature_importance_dict=None, input_features=None,feature_names=None):
    user_input_lower = user_input.lower()

    # Check for 'why' or 'reason' AFTER prediction
    if ("why" in user_input_lower or "reason" in user_input_lower):
        if prediction is not None and probability is not None and input_features is not None:
            return explain_prediction(probability, input_features, feature_importance_dict)
        else:
            return "I need a prediction first before I can explain it. Please submit the form."

    # General questions handling
    elif "what is customer churn" in user_input_lower:
        return "Customer churn is when a customer stops doing business with a company or stops using its services. It's a key metric in customer retention."

    elif "how does this app work" in user_input_lower or "how to use" in user_input_lower:
        return "This app predicts whether a customer is likely to churn. Fill in the customer details in the form and click Predict. I’ll explain the prediction if you ask why."

    elif "what can you do" in user_input_lower:
        return "I can explain churn predictions based on the model's results. Ask me 'why?' after submitting the form."

    elif "features" in user_input_lower:
       return f"The model uses the following features: {', '.join(feature_names)}."

    else:
        return "I'm still learning. Try asking me 'Why?' after submitting the prediction, or ask about customer churn or how the app works."
