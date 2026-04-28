import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.commentator import commentate, ask_race_question

st.set_page_config(
    page_title="PitWall AI — F1 Intelligence",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #050505 !important;
    color: #f0ede8 !important;
}

/* Hide streamlit default elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 2rem 2rem 2rem !important; max-width: 100% !important;}

/* HEADER BANNER */
.pw-header {
    background: linear-gradient(135deg, #0a0a0a 0%, #150005 100%);
    border-bottom: 1px solid rgba(232,0,29,0.3);
    padding: 28px 48px;
    margin: -1rem -2rem 2rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.pw-logo {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 0.2em;
    color: #f0ede8;
}
.pw-logo span { color: #e8001d; }
.pw-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.25em;
    color: rgba(240,237,232,0.4);
    text-transform: uppercase;
}
.pw-live {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #e8001d;
    border: 1px solid rgba(232,0,29,0.4);
    padding: 6px 14px;
    border-radius: 2px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { border-color: rgba(232,0,29,0.3); }
    50% { border-color: rgba(232,0,29,0.9); }
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 3px;
    padding: 24px 28px;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(232,0,29,0.3); }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    color: rgba(240,237,232,0.45);
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 36px;
    font-weight: 900;
    color: #e8001d;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: rgba(240,237,232,0.4);
    margin-top: 6px;
}

/* SECTION LABELS */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.35em;
    color: #e8001d;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.sec-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #f0ede8;
    margin-bottom: 24px;
    letter-spacing: -0.02em;
}
.sec-title span { color: rgba(240,237,232,0.35); }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(240,237,232,0.45) !important;
    padding: 14px 28px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #f0ede8 !important;
    border-bottom-color: #e8001d !important;
    background: transparent !important;
}

/* BUTTONS */
.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    background: #e8001d !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 12px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* SELECTBOX & INPUTS */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 2px !important;
    color: #f0ede8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* CHAT */
.stChatMessage {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 3px !important;
}

/* DIVIDER */
.pw-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,0,29,0.4), transparent);
    margin: 32px 0;
}

/* DRIVER ROW */
.driver-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: rgba(255,255,255,0.02);
    border-left: 2px solid transparent;
    border-radius: 2px;
    margin-bottom: 6px;
    transition: all 0.2s;
}
.driver-row:hover { border-left-color: #e8001d; background: rgba(232,0,29,0.04); }

/* RESULT BOX */
.result-box {
    background: rgba(232,0,29,0.06);
    border: 1px solid rgba(232,0,29,0.25);
    border-radius: 3px;
    padding: 24px;
    margin-top: 16px;
}
.result-time {
    font-family: 'Orbitron', sans-serif;
    font-size: 48px;
    font-weight: 900;
    color: #e8001d;
    line-height: 1;
}

/* COMMENTARY BOX */
.commentary-box {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #e8001d;
    border-radius: 0 3px 3px 0;
    padding: 20px 24px;
    font-style: italic;
    font-size: 15px;
    line-height: 1.8;
    color: rgba(240,237,232,0.7);
    min-height: 100px;
}

/* SPINNER */
.stSpinner > div { border-top-color: #e8001d !important; }

/* CAPTION */
.stCaption { color: rgba(240,237,232,0.4) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; letter-spacing: 0.15em !important; }

/* METRIC overrides */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: #e8001d !important;
    font-size: 32px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: rgba(240,237,232,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='rgba(240,237,232,0.7)', size=12),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickfont=dict(family='JetBrains Mono', size=10)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', tickfont=dict(family='JetBrains Mono', size=10)),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='JetBrains Mono', size=10)),
    colorway=['#e8001d','#ff4d6d','#ff8fa3','#ffb3c1','#ffd600','#ffffff','#aaaaaa'],
    margin=dict(l=10, r=10, t=40, b=10),
    title_font=dict(family='Orbitron', size=14, color='#f0ede8'),
)

RED = '#e8001d'
COLORS = {'SOFT': '#e8001d', 'MEDIUM': '#ffd600', 'HARD': '#f0ede8'}

# ── HEADER ──
st.markdown("""
<div class="pw-header">
  <div>
    <div class="pw-logo">PIT<span>WALL</span> AI</div>
    <div class="pw-tagline">F1 Intelligence Platform — Season 2024</div>
  </div>
  <div class="pw-live">● LIVE DATA</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ──
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("models/race_predictor.json")
    return model

@st.cache_data
def load_laps():
    if os.path.exists("data/laps.csv"):
        df = pd.read_csv("data/laps.csv")
        df = df[(df["lap_time_s"] > 60) & (df["lap_time_s"] < 300)]
        return df
    return pd.DataFrame()

model = load_model()
df = load_laps()

# ── TOP METRICS ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-label">Laps Analysed</div><div class="metric-value">22,126</div><div class="metric-sub">Real F1 telemetry</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-label">Model MAE</div><div class="metric-value">2.03s</div><div class="metric-sub">XGBoost Regressor</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-label">Races Covered</div><div class="metric-value">15</div><div class="metric-sub">2023 — 2024 season</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-label">AI Styles</div><div class="metric-value">3</div><div class="metric-sub">Brundle · Crofty · Analyst</div></div>', unsafe_allow_html=True)

st.markdown('<div class="pw-divider"></div>', unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs(["📊  Analytics", "⏱  Predictor", "🎙  Commentator", "❓  Ask AI"])

# ════════════════════════════════
# TAB 1 — ANALYTICS
# ════════════════════════════════
with tab1:
    st.markdown('<div class="sec-label">// Race Analytics</div><div class="sec-title">Live Race <span>Dashboard</span></div>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data loaded")
    else:
        col_l, col_r = st.columns([1.2, 1])

        with col_l:
            # Race selector
            gp = st.selectbox("Select Grand Prix", df["gp"].unique(), label_visibility="collapsed")
            race_df = df[df["gp"] == gp].copy()

            # Lap time chart — area style
            fig_lap = go.Figure()
            drivers = race_df["driver"].unique()[:6]
            palette = [RED, '#ff4d6d', '#ffd600', '#ffffff', '#aaaaaa', '#666666']
            for i, drv in enumerate(drivers):
                d = race_df[race_df["driver"]==drv].sort_values("lap")
                fig_lap.add_trace(go.Scatter(
                    x=d["lap"], y=d["lap_time_s"],
                    name=drv, mode='lines',
                    line=dict(color=palette[i % len(palette)], width=1.8),
                    fill='tozeroy' if i == 0 else 'none',
                    fillcolor='rgba(232,0,29,0.06)' if i == 0 else None,
                ))
            fig_lap.update_layout(
                **PLOT_THEME,
                title=f"Lap Times — {gp}",
                height=320,
                xaxis_title="Lap",
                yaxis_title="Time (s)",
            )
            st.plotly_chart(fig_lap, use_container_width=True)

            # Tyre box chart
            tyre_df = race_df[race_df["compound"].isin(["SOFT","MEDIUM","HARD"])]
            fig_tyre = go.Figure()
            for compound, color in COLORS.items():
                cd = tyre_df[tyre_df["compound"]==compound]["lap_time_s"]
                if len(cd) > 0:
                    fig_tyre.add_trace(go.Box(
                        y=cd, name=compound,
                        marker_color=color,
                        line_color=color,
                        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)',
                        boxmean=True,
                    ))
            fig_tyre.update_layout(
                **PLOT_THEME,
                title="Tyre Compound Performance",
                height=280,
                yaxis_title="Lap Time (s)",
                showlegend=False,
            )
            st.plotly_chart(fig_tyre, use_container_width=True)

        with col_r:
            # Stint analysis — scatter
            fig_stint = go.Figure()
            for compound, color in COLORS.items():
                cd = race_df[race_df["compound"]==compound]
                if len(cd) > 0:
                    fig_stint.add_trace(go.Scatter(
                        x=cd["tyre_life"], y=cd["lap_time_s"],
                        mode='markers', name=compound,
                        marker=dict(color=color, size=4, opacity=0.6),
                    ))
            fig_stint.update_layout(
                **PLOT_THEME,
                title="Tyre Age vs Lap Time",
                height=280,
                xaxis_title="Tyre Age (laps)",
                yaxis_title="Lap Time (s)",
            )
            st.plotly_chart(fig_stint, use_container_width=True)

            # Driver pace heatmap style — bar chart
            avg_pace = race_df.groupby("driver")["lap_time_s"].median().sort_values().head(8)
            fig_pace = go.Figure(go.Bar(
                x=avg_pace.values,
                y=avg_pace.index,
                orientation='h',
                marker=dict(
                    color=avg_pace.values,
                    colorscale=[[0, RED], [0.5, '#ff4d6d'], [1, '#ffb3c1']],
                    showscale=False,
                ),
                text=[f"{v:.3f}s" for v in avg_pace.values],
                textposition='outside',
                textfont=dict(family='JetBrains Mono', size=10, color='rgba(240,237,232,0.6)'),
            ))
            fig_pace.update_layout(
                **PLOT_THEME,
                title="Driver Median Pace",
                height=280,
                xaxis_title="Median Lap Time (s)",
            )
            st.plotly_chart(fig_pace, use_container_width=True)

            # Lap distribution histogram
            fig_hist = go.Figure(go.Histogram(
                x=race_df["lap_time_s"],
                nbinsx=30,
                marker_color=RED,
                opacity=0.7,
            ))
            fig_hist.update_layout(
                **PLOT_THEME,
                title="Lap Time Distribution",
                height=240,
                xaxis_title="Lap Time (s)",
                yaxis_title="Count",
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════
# TAB 2 — PREDICTOR
# ════════════════════════════════
with tab2:
    st.markdown('<div class="sec-label">// ML Model</div><div class="sec-title">Lap Time <span>Predictor</span></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        compound = st.selectbox("Tyre Compound", ["SOFT", "MEDIUM", "HARD"])
        tyre_life = st.slider("Tyre Age (laps)", 1, 50, 15)
        lap_num = st.slider("Lap Number", 1, 70, 30)
        stint = st.slider("Stint Number", 1, 4, 1)
        cmap = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}

        if st.button("▶  Run Prediction"):
            features = np.array([[tyre_life, lap_num, stint, cmap[compound], 5, 0, 2024]])
            pred = float(model.predict(features)[0])
            mins = int(pred // 60)
            secs = pred % 60
            st.markdown(f"""
            <div class="result-box">
                <div class="metric-label">Predicted Lap Time</div>
                <div class="result-time">{mins}:{secs:06.3f}</div>
                <div class="metric-sub" style="margin-top:8px">Model MAE ±1.44s · XGBoost · 6,745 training laps</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # Tyre degradation simulation chart
        laps_range = list(range(1, 51))
        deg_data = []
        for comp, enc, color in [("SOFT",2,RED),("MEDIUM",1,"#ffd600"),("HARD",0,"#f0ede8")]:
            preds = []
            for la in laps_range:
                feat = np.array([[la, 30, 1, enc, 5, 0, 2024]])
                preds.append(float(model.predict(feat)[0]))
            deg_data.append((comp, preds, color))

        fig_deg = go.Figure()
        for comp, preds, color in deg_data:
            fig_deg.add_trace(go.Scatter(
                x=laps_range, y=preds,
                name=comp, mode='lines',
                line=dict(color=color, width=2.5),
            ))
        fig_deg.update_layout(
            **PLOT_THEME,
            title="Tyre Degradation Curves — All Compounds",
            height=340,
            xaxis_title="Tyre Age (laps)",
            yaxis_title="Predicted Lap Time (s)",
        )
        st.plotly_chart(fig_deg, use_container_width=True)

        # Model performance bar
        st.markdown('<div class="metric-label" style="margin-top:16px">Model Metrics</div>', unsafe_allow_html=True)
        metrics = {"Accuracy": 98.4, "Training Coverage": 74, "Race Coverage": 40}
        fig_m = go.Figure(go.Bar(
            x=list(metrics.values()),
            y=list(metrics.keys()),
            orientation='h',
            marker_color=[RED, 'rgba(240,237,232,0.3)', 'rgba(240,237,232,0.3)'],
            text=[f"{v}%" for v in metrics.values()],
            textposition='outside',
            textfont=dict(family='JetBrains Mono', size=10, color='rgba(240,237,232,0.6)'),
        ))
        fig_m.update_layout(
            **PLOT_THEME,
            height=180,
            xaxis_range=[0, 120],
            showlegend=False,
        )
        st.plotly_chart(fig_m, use_container_width=True)

# ════════════════════════════════
# TAB 3 — COMMENTATOR
# ════════════════════════════════
with tab3:
    st.markdown('<div class="sec-label">// AI Commentary</div><div class="sec-title">Race <span>Commentator</span></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        style = st.selectbox("Commentary Style", ["brundle", "crofty", "analyst"],
            format_func=lambda x: {"brundle":"🎙 Martin Brundle","crofty":"📢 David Croft","analyst":"📊 Data Analyst"}[x])
        leader = st.text_input("Race Leader", "VER")
        driver = st.text_input("Driver in Focus", "LEC")
        gap = st.number_input("Gap to Leader (s)", 0.0, 60.0, 3.2)
        t_age = st.slider("Tyre Age", 1, 50, 18)
        weather = st.selectbox("Conditions", ["dry", "wet", "damp"])

        if st.button("▶  Generate Commentary"):
            with st.spinner("Generating..."):
                lap_data = {
                    "lap": 30, "leader": leader, "gap": gap,
                    "driver": driver, "compound": "SOFT",
                    "tyre_age": t_age, "last_lap": 91.4, "weather": weather
                }
                result = commentate(lap_data, style)
                st.session_state["commentary"] = result

    with col_r:
        st.markdown('<div class="metric-label" style="margin-bottom:12px">Generated Commentary</div>', unsafe_allow_html=True)
        commentary_text = st.session_state.get("commentary", "Click Generate to produce AI race commentary from live lap data...")
        st.markdown(f'<div class="commentary-box">{commentary_text}</div>', unsafe_allow_html=True)

        # Tyre radar chart
        fig_radar = go.Figure(go.Scatterpolar(
            r=[85, 72, 90, 68, 80],
            theta=['Grip', 'Durability', 'Pace', 'Temp Range', 'Wet Perf'],
            fill='toself',
            fillcolor='rgba(232,0,29,0.15)',
            line=dict(color=RED, width=2),
            name='SOFT'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[72, 85, 78, 80, 72],
            theta=['Grip', 'Durability', 'Pace', 'Temp Range', 'Wet Perf'],
            fill='toself',
            fillcolor='rgba(255,214,0,0.1)',
            line=dict(color='#ffd600', width=2),
            name='MEDIUM'
        ))
        fig_radar.update_layout(
            **PLOT_THEME,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], gridcolor='rgba(255,255,255,0.08)', tickfont=dict(size=9)),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                bgcolor='rgba(0,0,0,0)',
            ),
            title="Tyre Compound Characteristics",
            height=320,
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════
# TAB 4 — ASK AI
# ════════════════════════════════
with tab4:
    st.markdown('<div class="sec-label">// AI Assistant</div><div class="sec-title">Ask Anything <span>About F1</span></div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Welcome to PitWall AI! Ask me anything about F1 — race strategy, tyre degradation, lap times, or our 6,745-lap dataset from 2023–2024!"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.markdown('<div class="metric-label" style="margin:12px 0 8px">Quick questions</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    quick = None
    if col1.button("Monaco strategy"): quick = "Best tyre strategy for Monaco?"
    if col2.button("Tyre degradation"): quick = "How does tyre degradation work in F1?"
    if col3.button("Undercut vs overcut"): quick = "Explain the undercut vs overcut strategy"
    if col4.button("VER 2023 dominance"): quick = "Why did Verstappen dominate 2023?"

    prompt = st.chat_input("Ask about F1 strategy, tyres, drivers...") or quick

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = {
                    "dataset": "6745 laps from 6 races 2023-2024",
                    "races": ["Bahrain", "Saudi Arabia", "Australia", "Monaco"],
                    "model_mae": "1.44s XGBoost"
                }
                try:
                    from llm.rag import rag_answer
                    from groq import Groq
                    import os
                    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                    answer = rag_answer(prompt, groq_client)
                except Exception as e:
                    answer = ask_race_question(prompt, context)





                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
