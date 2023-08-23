import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle missing values using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Random Forest model
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Load the trained model (you don't need to load the notebook)
# Use 'model.pkl' instead of 'Diabetes_prediction.ipynb'
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title('Diabetes Prediction')
    
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
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Show the prediction result
    if prediction[0] == 1:
        st.error('The person is predicted to have diabetes.')
    else:
        st.success('The person is predicted to be diabetes-free.')

if __name__ == '__main__':
    main()



