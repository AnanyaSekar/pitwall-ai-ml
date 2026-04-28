# 🏎 PitWall AI — F1 Intelligence Platform
 
<div align="center">
**Real F1 telemetry · XGBoost ML · RAG Chatbot · LLM Commentary · Live Deployed**
 
[![🏎 Live Dashboard](https://img.shields.io/badge/🏎_Live_Dashboard-Streamlit-e8001d?style=for-the-badge)](https://pitwall-ai-ml-ananya.streamlit.app)
[![🌐 Landing Page](https://img.shields.io/badge/🌐_Landing_Page-Firebase-FF6F00?style=for-the-badge)](https://pitwall-ai-e6ac7.web.app)
[![⭐ GitHub](https://img.shields.io/badge/GitHub-AnanyaSekar-181717?style=for-the-badge&logo=github)](https://github.com/AnanyaSekar/pitwall-ai-ml)
 
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-189AB4?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Firebase](https://img.shields.io/badge/Firebase-Firestore-FFCA28?style=flat-square&logo=firebase&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-LLaMA_70B-00A67E?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange?style=flat-square)
 
</div>
---
 
## 🔗 Live Links
 
| | URL | What's there |
|---|---|---|
| 🌐 | [pitwall-ai-e6ac7.web.app](https://pitwall-ai-e6ac7.web.app) | 3D F1 landing page with Three.js |
| 📊 | [pitwall-ai-ml-ananya.streamlit.app](https://pitwall-ai-ml-ananya.streamlit.app) | Full ML dashboard |
 
---
 
## 🧠 What is PitWall AI?
 
PitWall AI is a **production-grade F1 intelligence platform** that combines real race telemetry, machine learning, and large language models. It was built entirely from scratch with free tools and is deployed live on the internet.
 
This is not a toy dataset project. Every lap, tyre, and sector time comes from **real Formula 1 races** via the FastF1 API.
 
```
"My XGBoost model would've recommended a different pit window
 for Leclerc in Monaco 2024 — and the data backs it up."
```
 
---
 
## 🚀 Features
 
### 📊 Race Analytics Dashboard
- Interactive lap time area charts across **15 real Grand Prix races**
- Tyre compound box plots — Soft vs Medium vs Hard performance
- Driver median pace comparison — horizontal bar chart
- Tyre age vs lap time scatter plot showing degradation in real data
- Lap time distribution histogram per race
### ⏱ ML Lap Time Predictor
- **XGBoost Regressor** trained on **22,126 real F1 laps**
- **MAE: 2.03 seconds** — predicts within 2.03s of real lap time
- Features: tyre compound, tyre age, lap number, stint, driver encoding, circuit, year
- Live tyre degradation curve simulation — all 3 compounds
- Experiment tracking with MLflow
### 🎙 AI Race Commentator
- **Groq LLaMA 3.3 70B** generates live race commentary from raw telemetry
- 3 styles: **Martin Brundle** (technical) · **David Croft** (dramatic) · **Data Analyst** (statistical)
- Feeds real lap data, gap, tyre age, weather into the LLM prompt
### 🔍 RAG Pipeline
- **ChromaDB** vector database storing all 22,094 laps as embeddings
- **sentence-transformers** (all-MiniLM-L6-v2) converts questions to vectors
- Retrieves the 15 most relevant real laps before generating any answer
- Answers grounded in actual telemetry — not general F1 knowledge
- Example: *"Leclerc ran 77.8s on lap 42 at Monaco 2023 on Hard tyres"* — from real data
### ❓ F1 AI Chatbot
- Ask anything about F1 — strategy, tyres, drivers, regulations, race history
- **Persistent chat memory** across the session using Streamlit session state
- Answers grounded in 22,094 real laps via RAG retrieval
---
 
## 📈 Model Performance
 
| Metric | Value |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Training laps** | 22,126 |
| **Races** | 15 GPs across 2022, 2023, 2024 |
| **MAE** | **2.03 seconds** |
| **RMSE** | **4.32 seconds** |
| **R² Score** | **0.9972** |
| **Cross-val R²** | **0.9831 ± 0.031 (5-fold)** |
| **Estimators** | 300 · Max depth 6 · LR 0.05 |
 
> Cross-validated R² of 0.98 confirms the model generalises well and is not overfitting.
> Baseline (predicting median lap time): MAE ~8.2s — our model beats baseline by **5.6×**
 
### 🔍 Feature Importance (SHAP)
 
![SHAP Feature Importance](models/shap_importance.png)
 
> **Lap number** is the strongest predictor — race situation matters more than tyre state alone. **Circuit** is second, confirming Monaco vs Bahrain dominate performance differences. **Tyre Age** and **Compound** follow, capturing degradation curves accurately.
 
---
 
## 🔬 Key Findings from the Data
 
- **Lap number beats tyre age** as the strongest predictor (SHAP) — race situation matters more than tyre state
- **Monaco laps average 78s** vs **Bahrain at 91s** — circuit characteristics dominate performance
- **Soft tyres degrade at ~0.045s/lap** vs Hard at ~0.013s/lap — 3.5× faster degradation
- **Verstappen's median pace at Bahrain 2023** was 0.4s faster than nearest rival
- Our model **beats naive baseline by 5.6×** — MAE 2.03s vs baseline 8.2s
---
 
## 🏗 System Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                          │
│   FastF1 API ──► Python ETL ──► Firebase Firestore      │
│   Ergast API ──► 22,126 laps · 15 GPs · 2022–2024      │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                      ML LAYER                           │
│   XGBoost Regressor · MAE 2.03s · MLflow tracking      │
│   SHAP explainability · Tyre degradation curves         │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     RAG LAYER                           │
│   ChromaDB · sentence-transformers · 22,094 embeddings  │
│   Vector search · Grounded LLM answers                  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     LLM LAYER                           │
│   Groq LLaMA 3.3 70B · Commentary · RAG chatbot        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   FRONTEND LAYER                        │
│   Streamlit Dashboard · Plotly Charts · Custom CSS      │
│   Orbitron + JetBrains Mono fonts · Dark F1 theme       │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  DEPLOYMENT LAYER                       │
│   Streamlit Cloud · Firebase Hosting · GitHub Actions   │
└─────────────────────────────────────────────────────────┘
```
 
---
 
## 📦 Tech Stack
 
| Layer | Technology | Why |
|---|---|---|
| **Data** | FastF1, Ergast API, Pandas | Real F1 telemetry, free |
| **ML** | XGBoost, Scikit-learn | Best gradient boosting for tabular data |
| **Explainability** | SHAP | Feature importance — shows what drives predictions |
| **Experiment tracking** | MLflow | Log every model run, compare metrics |
| **RAG** | ChromaDB, sentence-transformers | Ground LLM answers in real lap data |
| **LLM** | Groq API (LLaMA 3.3 70B) | Free, fastest inference available |
| **Frontend** | Streamlit, Plotly | Rapid deployment, interactive charts |
| **Database** | Firebase Firestore | Free NoSQL, real-time, no server needed |
| **Hosting** | Streamlit Cloud + Firebase | 100% free deployment |
| **3D Landing** | Three.js | Cinematic F1 car hero section |
| **CI/CD** | GitHub Actions | Auto deploy on every push |
 
---
 
## 🗂 Project Structure
 
```
pitwall-ai/
├── data/
│   ├── ingest.py              # FastF1 → Firestore ETL pipeline
│   └── laps.csv               # 22,126 laps exported for cloud deploy
├── models/
│   ├── race_predictor.py      # XGBoost training + MLflow logging
│   ├── evaluate.py            # R², MAE, RMSE, cross-validation
│   ├── shap_analysis.py       # SHAP feature importance
│   ├── shap_importance.png    # SHAP chart
│   └── race_predictor.json    # Trained model (300 estimators)
├── llm/
│   ├── commentator.py         # Groq LLaMA commentary + F1 chatbot
│   └── rag.py                 # ChromaDB RAG pipeline — 22,094 laps indexed
├── frontend/
│   └── app.py                 # Streamlit dashboard (custom dark theme)
├── public/
│   └── index.html             # Three.js 3D F1 landing page
├── .github/
│   └── workflows/             # GitHub Actions CI/CD
└── requirements.txt
```
 
---
 
## ⚡ Run Locally
 
```bash
# Clone the repo
git clone https://github.com/AnanyaSekar/pitwall-ai-ml.git
cd pitwall-ai-ml
 
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
 
# Install dependencies
pip install -r requirements.txt
 
# Add API key
echo "GROQ_API_KEY=your_groq_key_here" > .env
 
# Build RAG vector database (first time only)
python3 llm/rag.py
 
# Run dashboard
streamlit run frontend/app.py
```
 
> Get a free Groq API key at [console.groq.com](https://console.groq.com)
 
---
 
## 💡 Key Technical Decisions
 
**Why XGBoost over a neural network?**
Tabular F1 data has strong non-linear relationships between tyre age and lap time, but the dataset (22,126 rows) benefits more from gradient boosting than deep learning at this scale. XGBoost is also fully interpretable via SHAP — you can explain every prediction, which matters in a domain where decisions have real consequences.
 
**Why RAG instead of just prompting the LLM?**
Without RAG, the chatbot answers from general F1 knowledge — it cannot tell you Leclerc's actual lap 42 time at Monaco 2023. With RAG, we embed all 22,094 laps into ChromaDB, retrieve the 15 most relevant laps for each question, and ground the LLM's answer in real telemetry. The answer "Leclerc ran 77.8s on lap 42 at Monaco 2023 on Hard tyres" comes from your actual dataset — not a hallucination.
 
**Why Groq instead of OpenAI?**
Groq's LLaMA 3.3 70B is completely free with generous rate limits, and inference is faster than GPT-4. Free + fast beats expensive + marginal quality gains for a production portfolio project.
 
**Why Firestore instead of PostgreSQL?**
No server to manage, free tier covers 1GB + 50k reads/day, and real-time updates are built in. Perfect for a project at this scale.
 
---
 
## 🔬 Model Explainability
 
The SHAP analysis revealed that **lap number** is the strongest predictor of lap time — more important than tyre age or compound. This is a genuine insight: early laps have fuel weight effects and tyre warm-up phases, while late laps have degradation and track evolution. The race situation captures more information than just the tyre state.
 
Circuit encoding is the second most important feature, confirming that Monaco (78s laps) and Bahrain (91s laps) are fundamentally different prediction problems — the model correctly learns this from data.
 
---
 
## 👩‍💻 About
 
Built by **Ananya Sekar** — a data science and AI enthusiast who loves Formula 1.
 
This project was built from scratch — data pipeline, ML model, SHAP explainability, RAG pipeline, LLM integration, frontend, and deployment — using only free tools.
 
---
 
<div align="center">
**⭐ Star this repo if you found it useful!**
 
*Built with ❤️ and lots of F1 data*
 
</div>
 
