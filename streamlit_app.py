import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Train your model
model =


dataset = pd.read_csv('diabetes.csv')
dataset.shape
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")



dataset.corr()

# check if any null value is present
dataset.isnull().values.any()


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, y_train.ravel())
(predict_train_data := random_forest_model.predict(X_test))
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
inputdata1=np.asarray(inputdata)
inputdata2=inputdata1.reshape(1,-1)
predict=classifier.predict(inputdata2)

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)



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
    
    # Load the trained model
    with open('Diabetes_prediction.ipynb', 'rb') as file:
        model = pickle.load(file)
    
    # Predict
    if st.button('Predict'):
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'Age': [age],
            'BMI': [bmi]
            # Add other feature columns with the corresponding input values
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


