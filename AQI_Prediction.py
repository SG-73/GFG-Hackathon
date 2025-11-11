# Air Quality Index (AQI) Prediction App
# Built using Streamlit and a pre-trained XGBoost model

# Importing required libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the saved model and metadata

# 'city.pkl' - list of all city names (for dropdown selection)
# 'col.pkl'  - list of all input feature columns used during training
# 'aqi.pkl'  - trained ML model (XGBoost / RandomForest / etc.)
c = pickle.load(open('city.pkl', 'rb'))
col = pickle.load(open('col.pkl', 'rb'))
xgb = pickle.load(open('aqi.pkl', 'rb'))

# Function to predict AQI category
def predict_AQI(data):
    """
    Predicts the Air Quality Index (AQI) category based on user input.

    Parameters:
        data (list): User-provided inputs (city and pollutant values)
    Returns:
        str: AQI category ('GOOD', 'SATISFACTORY', 'MODERATE', 'POOR', etc.)
    """

    # Initialize a zero array with length equal to number of model features
    x = np.zeros(len(col))

    # Assign pollutant input values to feature array (numeric values only)
    # Each pollutant corresponds to one column index
    x[0] = data[1]
    x[1] = data[2]
    x[2] = data[3]
    x[3] = data[4]
    x[4] = data[5]
    x[5] = data[6]
    x[6] = data[7]
    x[7] = data[8]
    x[8] = data[9]
    x[9] = data[10]
    x[10] = data[11]
    x[11] = data[12]

    # One-hot encode the city column
    city = data[0]
    df = pd.DataFrame([x], columns=col)
    city_index = np.where(df.columns == city)
    x[city_index] = 1

    # Reshape feature vector for model input
    x_reshape = x.reshape(1, -1)

    # Predict AQI using the trained model
    pred = xgb.predict(x_reshape)
    pred = np.round(pred, 2)

    # Categorize AQI based on prediction value
    if pred <= 50:
        return 'GOOD'
    elif pred <= 100:
        return 'SATISFACTORY'
    elif pred <= 200:
        return 'MODERATE'
    elif pred <= 300:
        return 'POOR'
    elif pred <= 400:
        return 'VERY POOR'
    else:
        return 'SEVERE'

# Streamlit UI definition
def main():
    """Main function to render the Streamlit web app UI."""

    # App title
    st.title('🌆 Air Quality Index (AQI) Prediction Web App')
    st.write('This app predicts the air quality level based on pollutant concentrations.')

    # Input section — allows user to select city and enter pollutant values
    City = st.selectbox('Select City', c)
    PM2_5 = st.text_input('PM2.5 Level (μg/m³)')
    PM10 = st.text_input('PM10 Level (μg/m³)')
    NO = st.text_input('NO Level (μg/m³)')
    NO2 = st.text_input('NO₂ Level (μg/m³)')
    NOx = st.text_input('NOx Level (μg/m³)')
    NH3 = st.text_input('NH₃ Level (μg/m³)')
    CO = st.text_input('CO Level (mg/m³)')
    SO2 = st.text_input('SO₂ Level (μg/m³)')
    O3 = st.text_input('O₃ Level (μg/m³)')
    Benzene = st.text_input('Benzene Level (μg/m³)')
    Toluene = st.text_input('Toluene Level (μg/m³)')
    Xylene = st.text_input('Xylene Level (μg/m³)')

    result = ''  # Variable to store prediction result

    # When user clicks 'Calculate', perform prediction
    if st.button('Calculate'):
        result = predict_AQI([
            City, PM2_5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
        ])
    
    # Display prediction result
    st.success(f'Predicted AQI Category: {result}')

    # Optional: print all columns in the console (for debugging)
    print(col)

# Run the app
if __name__ == '__main__':
    main()
