import pickle
import pandas as pd

from src.preprocess import preprocess_data


def predict_price(input_data: dict):
    # Load artifacts
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_columns = pickle.load(open("features.pkl", "rb"))

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Preprocess
    X_processed = preprocess_data(
        df,
        training=False,
        scaler=scaler,
        feature_columns=feature_columns
    )

    # Predict
    prediction = model.predict(X_processed)
    return prediction[0]


if __name__ == "__main__":
    
    sample_input = {
        "Year": 2017,
        "Present_Price": 3.6,
        "Kms_Driven": 2135,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }

    price = predict_price(sample_input)
    print(f"Predicted Selling Price: {price:.2f}")
