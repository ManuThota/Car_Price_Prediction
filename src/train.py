import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.data_loader import load_data
from src.preprocess import preprocess_data


def train():
    # Load data
    df = load_data("data/car_price_data.csv")

    # Split target
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess
    X_train_scaled, scaler, feature_columns = preprocess_data(X_train, training=True)
    X_test_scaled = preprocess_data(X_test, training=False, scaler=scaler, feature_columns=feature_columns)

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    print(f"Training completed | R2 Score: {r2:.4f}")

    # Save artifacts
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(feature_columns, open("features.pkl", "wb"))

    print("Model & scaler saved successfully")


if __name__ == "__main__":
    train()
