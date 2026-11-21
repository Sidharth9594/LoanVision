
import joblib
from feature_engineering import add_features

# Load trained model and preprocessing pipeline
model = joblib.load("models/trained_model.pkl")
pipeline = joblib.load("models/preprocessing_pipeline.pkl")


def predict_loan_default(applicant_data):

    # Convert applicant data into DataFrame
    import pandas as pd

    input_data = pd.DataFrame([applicant_data])

    # Generate required features
    input_data = add_features(input_data)

    # Apply preprocessing
    processed_data = pipeline.transform(input_data)

    # Predict default probability
    probability = model.predict_proba(processed_data)[0][1]

    # Predict class
    prediction = model.predict(processed_data)[0]

    # Determine risk category using project thresholds
    if probability < 0.0995131773:
        risk = "Low"
    elif probability < 0.2533386081:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk": risk
    }
