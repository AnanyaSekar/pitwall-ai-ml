import streamlit as st
import plotly.express as px
import pandas as pd
import xgboost as xgb
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.commentator import commentate, ask_race_question

st.set_page_config(page_title="PitWall AI", page_icon="🏎", layout="wide")

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("models/race_predictor.json")
    return model

@st.cache_data
def load_laps():
    if os.path.exists("data/laps.csv"):
        return pd.read_csv("data/laps.csv")
    return pd.DataFrame()

model = load_model()
st.title("🏎 PitWall AI — F1 Intelligence Platform")
st.caption("Real F1 data · XGBoost ML · AI Commentary")

tab1,tab2,tab3,tab4 = st.tabs(["📊 Analytics","⏱ Predictor","🎙 Commentator","❓ Ask AI"])

with tab1:
    st.subheader("Race Analytics")
    df = load_laps()
    if df.empty:
        st.warning("No data loaded")
    else:
        col1,col2,col3 = st.columns(3)
        col1.metric("Total Laps", f"{len(df):,}")
        col2.metric("Races", df["gp"].nunique())
        col3.metric("Drivers", df["driver"].nunique())
        gp = st.selectbox("Select race", df["gp"].unique())
        race_df = df[df["gp"]==gp]
        fig = px.line(race_df.sort_values("lap"), x="lap", y="lap_time_s",
                      color="driver", title=f"Lap times — {gp}")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(race_df, x="compound", y="lap_time_s",
                      color="compound", title=f"Tyre performance — {gp}")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Lap Time Predictor")
    st.caption("XGBoost model — MAE 1.44s on 6,745 real laps")
    col1,col2,col3 = st.columns(3)
    tyre_life = col1.slider("Tyre age",1,50,15)
    lap_num   = col2.slider("Lap number",1,70,30)
    stint     = col3.slider("Stint",1,4,1)
    compound  = st.selectbox("Compound",["SOFT","MEDIUM","HARD"])
    cmap = {"SOFT":2,"MEDIUM":1,"HARD":0}
    if st.button("Predict", type="primary"):
        pred = float(model.predict(np.array([[tyre_life,lap_num,stint,cmap[compound],5,0,2024]]))[0])
        st.success(f"Predicted: {int(pred//60)}:{pred%60:06.3f}")

with tab3:
    st.subheader("AI Race Commentator")
    style  = st.selectbox("Style",["brundle","crofty","analyst"])
    col1,col2 = st.columns(2)
    leader = col1.text_input("Leader","VER")
    driver = col2.text_input("Driver","LEC")
    gap    = col1.number_input("Gap (s)",0.0,60.0,3.2)
    t_age  = col2.slider("Tyre age",1,50,18)
    if st.button("Generate commentary", type="primary"):
        with st.spinner("Generating..."):
            result = commentate({"lap":30,"leader":leader,"gap":gap,
                                 "driver":driver,"compound":compound,
                                 "tyre_age":t_age,"last_lap":91.4,"weather":"dry"}, style)
            st.info(result)

with tab4:
    st.subheader("Ask the AI anything about F1")
    q = st.text_input("Question","Why would a driver pit early on lap 20?")
    if st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            st.write(ask_race_question(q, {"race":"Monaco","lap":20,"leader":"VER"}))
