import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

## load the model
model = tf.keras.models.load_model('model.h5')

## load the pickle files
with open('labelEncoder_gender.pkl', 'rb') as f:
    labelEncoder_gender = pickle.load(f)

with open('oneHotEncode_Geography.pkl', 'rb') as f:
    oneHotEncode_Geography = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will churn or not.")
geography = st.selectbox('Geography',oneHotEncode_Geography.categories_[0])
gender = st.selectbox('Gender',labelEncoder_gender.classes_)
age = st.slider('Age',18,99)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
creditScore = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
tenure = st.slider('Tenure',0,10)
numOfProducts = st.slider('Number of Products',1,4)
hasCrCard = st.selectbox('Has Credit Card',[0, 1])
isActiveMember = st.selectbox('Is Active Member',[0, 1])
estimatedSalary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

if st.button('Predict'):
    ## prepare the input data
    input_data = pd.DataFrame({
        'RowNumber': [1],  # Dummy RowNumber
        'CustomerId': [12345678],  # Dummy CustomerId
        'CreditScore': [creditScore],
        'Gender': [labelEncoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [numOfProducts],
        'HasCrCard': [hasCrCard],
        'IsActiveMember': [isActiveMember],
        'EstimatedSalary': [estimatedSalary]
    })

    # Encode Geography
    geography_encoded = oneHotEncode_Geography.transform(pd.DataFrame({'Geography': [geography]})).toarray()
    geography_df = pd.DataFrame(geography_encoded, columns=oneHotEncode_Geography.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data, geography_df], axis=1)

    #scale the input data
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    churn_prob = prediction[0][0]
    if churn_prob > 0.5:
        st.error(f'The customer is likely to churn with a probability of {churn_prob:.2f}')
    else:
        st.success(f'The customer is unlikely to churn with a probability of {churn_prob:.2f}')
