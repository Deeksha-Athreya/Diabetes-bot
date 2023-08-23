import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)


st.title('Diabetes Prediction App')


uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    predictions = model.predict(data)
    st.write(predictions)



