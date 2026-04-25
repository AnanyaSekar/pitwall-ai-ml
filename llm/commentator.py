import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", "")
        except:
            pass
    return Groq(api_key=key)

STYLES = {
    "brundle": "You are Martin Brundle, F1 commentator. Technical, calm, insightful. One paragraph only.",
    "crofty":  "You are David Croft, F1 commentator. Enthusiastic, dramatic, energetic. One paragraph only.",
    "analyst": "You are an F1 data scientist. Precise, statistical, factual. One paragraph only."
}

def commentate(lap_data: dict, style: str = "brundle") -> str:
    client = get_client()
    prompt = f"""
Race situation at lap {lap_data.get('lap', '?')}:
- Leader: {lap_data.get('leader', '?')} ({lap_data.get('gap', '?')}s ahead)
- Driver: {lap_data.get('driver', '?')}
- Tyre: {lap_data.get('compound', '?')}, age {lap_data.get('tyre_age', '?')} laps
- Last lap: {lap_data.get('last_lap', '?')}s
- Weather: {lap_data.get('weather', 'dry')}
Give one paragraph of live race commentary.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": STYLES.get(style, STYLES["brundle"])},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

def ask_race_question(question: str, context: dict) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert F1 analyst. Be specific and insightful."},
            {"role": "user", "content": f"Race data: {context}\n\nQuestion: {question}"}
        ],
        max_tokens=400
    )
    return response.choices[0].message.content
