import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, training=True, scaler=None, feature_columns=None):
    """
    Preprocess data for training and prediction
    """

    #  DROP STRING / NON-NUMERIC COLUMNS
    df = df.drop(columns=["Car_Name", "Selling_Price"], errors="ignore")

    # One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=["Fuel_Type", "Seller_Type", "Transmission"],
        drop_first=True
    )

    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        feature_columns = df.columns
        return X_scaled, scaler, feature_columns
    else:
        # Align columns with training data
        df = df.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(df)
        return X_scaled
