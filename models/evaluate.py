import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def load_data():
    docs = db.collection("laps").stream()
    df = pd.DataFrame([d.to_dict() for d in docs])
    le_c = LabelEncoder()
    le_d = LabelEncoder()
    le_g = LabelEncoder()
    df["compound_enc"] = le_c.fit_transform(df["compound"])
    df["driver_enc"]   = le_d.fit_transform(df["driver"])
    df["gp_enc"]       = le_g.fit_transform(df["gp"])
    return df

df = load_data()
features = ["tyre_life","lap","stint","compound_enc","driver_enc","gp_enc","year"]
X = df[features].fillna(0)
y = df["lap_time_s"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8)
model.fit(X_train,y_train)
preds = model.predict(X_test)

mae  = mean_absolute_error(y_test, preds)
r2   = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"\n{'='*40}")
print(f"MAE:  {mae:.4f}s")
print(f"RMSE: {rmse:.4f}s")
print(f"R2:   {r2:.4f}")
print(f"CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*40}\n")
