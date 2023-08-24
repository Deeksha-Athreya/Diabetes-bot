import numpy as np
import pandas as pd
import pickle
import streamlit as st




def main():
    st.title('Diabetes Prediction')

    # Load the trained model
    with open('classifier.pkl', 'rb') as file:
        model = pickle.load(file)

    # Input form
    pregnancies = st.slider('Number of Pregnancies', 0, 17, 0)
    glucose = st.number_input('Glucose', value=0)
    blood_pressure = st.number_input('Blood Pressure', value=0)
    skin_thickness = st.number_input('Skin Thickness', value=0)
    insulin = st.number_input('Insulin', value=0)
    age = st.slider('Age', 1, 100, 25)
    bmi = st.slider('BMI', 10.0, 60.0, 25.0)

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'Age': [age],
        'BMI': [bmi]
    })

 
    prediction = model.predict(input_data)

  
    if prediction[0] == 1:
        st.error('The person is predicted to have diabetes.')
    else:
        st.success('The person is predicted to be diabetes-free.')

if __name__ == '__main__':
    main()
