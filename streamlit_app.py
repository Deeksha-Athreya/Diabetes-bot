import streamlit as st
import pandas as pd
import joblib

def main():
    st.title('Diabetes Prediction')
    
   
    pregnancies = st.slider('Number of Pregnancies', 0, 17, 0)
    age = st.slider('Age', 1, 100, 25)
    bmi = st.slider('BMI', 10.0, 60.0, 25.0)
    
   
    model = joblib.load('model.joblib')
    
    # Predict
    if st.button('Predict'):
        
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
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

