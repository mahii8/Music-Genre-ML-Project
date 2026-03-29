import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the Light version of the model and scaler
try:
    model = joblib.load('music_model_light.pkl')
    scaler = joblib.load('scaler_light.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")

# 2. UI Header
st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵")
st.title("🎵 Music Genre Predictor")
st.markdown("Adjust the song characteristics below to predict the genre.")

st.divider()

# 3. User Inputs (Organized into 3 columns for 13 features)
col1, col2, col3 = st.columns(3)

with col1:
    popularity = st.slider("Popularity", 0, 100, 50)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    duration_ms = st.number_input("Duration (ms)", value=200000)
    valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)

with col2:
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1)
    key = st.number_input("Key (0-11)", 0, 11, 5)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)

with col3:
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    mode = st.selectbox("Mode", options=[0, 1], help="0 for Minor, 1 for Major")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    tempo = st.number_input("Tempo (BPM)", 40, 250, 120)

# 4. Prediction Logic
if st.button("Predict Genre", type="primary", use_container_width=True):
    # CRITICAL: These MUST stay in the exact order you gave me
    feature_names = [
        'popularity', 'acousticness', 'danceability', 'duration_ms', 
        'energy', 'instrumentalness', 'key', 'liveness', 
        'loudness', 'mode', 'speechiness', 'tempo', 'valence'
    ]
    
    # Create DataFrame to maintain feature names and order
    input_df = pd.DataFrame([[
        popularity, acousticness, danceability, duration_ms, 
        energy, instrumentalness, key, liveness, 
        loudness, mode, speechiness, tempo, valence
    ]], columns=feature_names)
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    st.success(f"### 🎧 Result: This song is likely **{prediction[0]}**")
    st.balloons()

st.divider()
st.caption("Music Genre Classification Project | Portfolio Piece")
