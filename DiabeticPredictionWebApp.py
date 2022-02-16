import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open ('E:/My work/work/Diabetes Pridiction/trained_diabetic_model.sav','rb'))

# creating a function for Prediction
def diabetes_prediction(input_data):
 
    # transforming this input list into numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # predicting
   
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return"The person is not diabetic"
    else:
        return"The person is diabetic"
        
        
def main():
    #giving a title
    st.title("Diabetic Prediction Web App")
    
    #getting the data from the user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure value")
    SkinThickness = st.text_input("SkinThickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
    Age = st.text_input("Age of person")
    
    
    #code for the Prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    

if __name__ =='__main__':
    main()
    
    
    
    