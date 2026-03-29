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
st.set_page_config(page_title="Music Genre AI", page_icon="🎵", layout="wide")
st.title("🎵 Music Genre Predictor")
st.write("This AI analyzes audio features to predict the musical genre.")

st.divider()

# 3. User Inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Vibe")
    popularity = st.slider("Popularity (0-100)", 0, 100, 50)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)

with col2:
    st.subheader("Technical")
    tempo = st.number_input("Tempo (BPM)", 40, 250, 120)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    duration_ms = st.number_input("Duration (ms)", value=200000)
    key = st.number_input("Key (0-11)", 0, 11, 5)
    mode = st.selectbox("Mode", options=[0, 1], format_func=lambda x: "Minor (1)" if x == 1 else "Major (0)")

with col3:
    st.subheader("Texture")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)

# 4. Prediction Logic
if st.button("Predict Genre", type="primary", use_container_width=True):
    # Order must match X_train columns exactly
    feature_names = ['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    
    input_df = pd.DataFrame([[
        popularity, acousticness, danceability, duration_ms, 
        energy, instrumentalness, key, liveness, 
        loudness, mode, speechiness, tempo, valence
    ]], columns=feature_names)
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    prediction_numeric = model.predict(input_scaled)[0]
    
    # --- YOUR UPDATED GENRE MAP ---
    genre_map = {
        0: "Electronic",
        1: "Anime",
        2: "Jazz",
        3: "Alternative",
        4: "Country",
        5: "Rap",
        6: "Blues",
        7: "Rock",
        8: "Classical",
        9: "Hip-Hop"
    }
    
    final_genre = genre_map.get(prediction_numeric, f"Unknown ({prediction_numeric})")
    
    st.success(f"### 🎧 Prediction: **{final_genre}**")
    st.balloons()

st.divider()
st.caption("Music Genre Classification Project | Built with Scikit-Learn & Streamlit")
