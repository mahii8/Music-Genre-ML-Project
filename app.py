import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the "Light" version of the model and scaler

try:
    model = joblib.load('music_model_light.pkl')
    scaler = joblib.load('scaler_light.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure music_model_light.pkl and scaler_light.pkl are in the repository.")

# 2. UI Header
st.set_page_config(page_title="Music Genre Predictor", page_icon="🎵")
st.title("🎵 Music Genre Predictor")
st.markdown("""
This app uses a **Random Forest Classifier** to predict the genre of a song based on its audio features. 
Adjust the sliders below and click **Predict** to see the AI's result!
""")

st.divider()

# 3. User Inputs (Left and Right columns for better layout)
col1, col2 = st.columns(2)

with col1:
    popularity = st.slider("Popularity (0-100)", 0, 100, 50)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.5)

with col2:
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    tempo = st.number_input("Tempo (BPM)", min_value=40, max_value=250, value=120)

# 4. Prediction Logic
if st.button("Predict Genre", type="primary", use_container_width=True):
    # Arrange inputs in the exact order the model expects
    input_data = np.array([[
        popularity, acousticness, danceability, energy, 
        instrumentalness, loudness, speechiness, tempo, valence
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Show result
    st.success(f"### 🎧 Result: This song is likely **{prediction[0]}**")
    st.balloons()

st.divider()
st.caption("Developed by [Your Name] | Built with Streamlit & Scikit-learn")
