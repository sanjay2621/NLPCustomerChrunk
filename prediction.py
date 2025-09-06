import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

## load the model
model = load_model('model.h5')

## load the pickle files

with open('labelEncoder_gender.pkl', 'rb') as f:
    labelEncoder_gender = pickle.load(f)

with open('oneHotEncode_Geography.pkl', 'rb') as f:
    oneHotEncode_Geography = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

input_df = pd.DataFrame([input_data])

# Preprocess the input data

# Encode Gender
input_df['Gender'] = labelEncoder_gender.transform(input_df['Gender'])
# Encode Geography
geography_encoded = oneHotEncode_Geography.transform(input_df[['Geography']]).toarray()
geography_df = pd.DataFrame(geography_encoded, columns=oneHotEncode_Geography.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df, geography_df], axis=1)

# Add missing columns expected by scaler with default values
for col in scaler.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
# Ensure column order matches scaler
input_df = input_df[scaler.feature_names_in_]
# Drop unnecessary columns before prediction


# Scale the numeric features
input_scaled = scaler.transform(input_df)

# Make prediction
predictions = model.predict(input_scaled)

## prediction probabilities

prediction_probabilities = predictions[0][0]
print("Prediction Probabilities:")
print(prediction_probabilities)

if prediction_probabilities > 0.5:
    print("Customer is likely to churn.")
else:
    print("Customer is not likely to churn.")
