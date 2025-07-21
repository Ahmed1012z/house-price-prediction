import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title("🏠 California House Price Prediction")

st.markdown("Enter the house features below to predict the price (in 100,000s USD):")

# User Inputs
med_inc = st.number_input("Median Income (MedInc)", min_value=0.0, max_value=20.0, value=3.0)
house_age = st.number_input("House Age", min_value=1.0, max_value=100.0, value=20.0)
avg_rooms = st.number_input("Average Number of Rooms (AveRooms)", min_value=1.0, max_value=50.0, value=5.0)
avg_bedrooms = st.number_input("Average Number of Bedrooms (AveBedrms)", min_value=1.0, max_value=10.0, value=1.0)
population = st.number_input("Population", min_value=1.0, max_value=40000.0, value=1000.0)
avg_occup = st.number_input("Average Occupancy (AveOccup)", min_value=0.5, max_value=100.0, value=3.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.0)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-119.0)

# Prediction Button
if st.button("Predict Price"):
    # Prepare input and apply scaling
    input_data = np.array([[med_inc, house_age, avg_rooms, avg_bedrooms,
                            population, avg_occup, latitude, longitude]])
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    st.success(f"💰 Estimated House Price: **${prediction[0]:.2f} × 100,000**")
