import numpy as np
import pandas as pd
import streamlit as st
import pickle

c = pickle.load(open('city.pkl', 'rb'))
col = pickle.load(open('col.pkl', 'rb'))
xgb = pickle.load(open('aqi.pkl', 'rb'))

def predict_AQI(data):

    x = np.zeros(len(col))
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

    city = data[0]
    df = pd.DataFrame([x], columns=col)
    city_index = np.where(df.columns==city)
    x[city_index] = 1

    x_reshape = x.reshape(1, -1)
    pred = xgb.predict(x_reshape)
    pred = np.round(pred, 2)

    if (pred <= 50):
        return 'GOOD'
    elif (pred <= 100):
        return 'SATISFACTORY'
    elif (pred <= 200):
        return 'MODERATE'
    elif (pred <= 300):
        return 'POOR'
    elif (pred <= 400):
        return 'VERY POOR'
    else:
        return 'SEVERE'

def main():
    st.title('AQI Prediction Web App')

    City = st.selectbox('Name of City', c)
    PM2_5 = st.text_input('PM2.5 Level')
    PM10 = st.text_input('PM10 Level')
    NO = st.text_input('NO Level')
    NO2 = st.text_input('NO2 Level')
    NOx = st.text_input('NOx Level')
    NH3 = st.text_input('NH3 Level')
    CO = st.text_input('CO Level')
    SO2 = st.text_input('SO2 Level')
    O3 = st.text_input('O3 Level')
    Benzene = st.text_input('Benzene Level')
    Toluene = st.text_input('Toluene Level')
    Xylene = st.text_input('Xylene Level')

    result = ''

    if st.button('Calculate'):
        result = predict_AQI([City, PM2_5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene])
    st.success(result)
    print(col)

if __name__ == '__main__':
    main()