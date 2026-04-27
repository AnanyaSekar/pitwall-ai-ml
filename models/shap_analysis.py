import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

docs = db.collection("laps").stream()
df = pd.DataFrame([d.to_dict() for d in docs])
le_c = LabelEncoder()
le_d = LabelEncoder()
le_g = LabelEncoder()
df["compound_enc"] = le_c.fit_transform(df["compound"])
df["driver_enc"]   = le_d.fit_transform(df["driver"])
df["gp_enc"]       = le_g.fit_transform(df["gp"])

features = ["tyre_life","lap","stint","compound_enc","driver_enc","gp_enc","year"]
X = df[features].fillna(0)
y = df["lap_time_s"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = xgb.XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05)
model.fit(X_train,y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:500])

plt.figure(figsize=(10,6))
plt.style.use('dark_background')
shap.summary_plot(shap_values, X_test[:500],
    feature_names=["Tyre Age","Lap","Stint","Compound","Driver","Circuit","Year"],
    show=False, plot_type="bar")
plt.title("PitWall AI — Feature Importance (SHAP)", color='white', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("models/shap_importance.png", dpi=150, bbox_inches='tight',
            facecolor='#050505', edgecolor='none')
print("SHAP plot saved!")
