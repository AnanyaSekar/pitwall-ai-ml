import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import mlflow
import mlflow.xgboost

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def load_data():
    print("Loading data from Firestore...")
    docs = db.collection("laps").stream()
    rows = [d.to_dict() for d in docs]
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} laps")
    return df

def train():
    df = load_data()
    le_compound = LabelEncoder()
    le_driver   = LabelEncoder()
    le_gp       = LabelEncoder()
    df["compound_enc"] = le_compound.fit_transform(df["compound"])
    df["driver_enc"]   = le_driver.fit_transform(df["driver"])
    df["gp_enc"]       = le_gp.fit_transform(df["gp"])

    features = ["tyre_life","lap","stint","compound_enc","driver_enc","gp_enc","year"]
    X = df[features].fillna(0)
    y = df["lap_time_s"]

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("pitwall-race-predictor")
    with mlflow.start_run():
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=50)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mlflow.log_metric("mae_seconds", mae)
        print(f"\nModel MAE: {mae:.3f} seconds")
        model.save_model("models/race_predictor.json")
        print("Model saved!")
    return model, mae

if __name__ == "__main__":
    model, mae = train()
    print(f"\nDone! Predicts lap times within {mae:.2f}s on average.")
