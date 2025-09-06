import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

data = pd.read_csv('Churn_Modelling.csv')
data.head()

## lets Encode the Categorical Features
labelEncoder_gender = LabelEncoder()
data['Gender'] = labelEncoder_gender.fit_transform(data['Gender'])
##print(data.head())

## Lets Encode the Geography
oneHotEncode_Geography = OneHotEncoder()
geography_encoded = oneHotEncode_Geography.fit_transform(data[['Geography']]).toarray()
geographyDF = pd.DataFrame(geography_encoded, columns=oneHotEncode_Geography.get_feature_names_out(['Geography']))
## Combine the DataFrames
data = pd.concat([data, geographyDF], axis=1)
data.drop('Geography', axis=1, inplace=True)

## print(data.head())

## save the pickle file future use

with open('data_preprocessed.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('labelEncoder_gender.pkl', 'wb') as f:
    pickle.dump(labelEncoder_gender, f)

with open('oneHotEncode_Geography.pkl', 'wb') as f:
    pickle.dump(oneHotEncode_Geography, f)

## devide the data into X and Y

X = data.drop(['Exited', 'Surname'], axis=1)
y = data['Exited']
numeric_cols = X.select_dtypes(include=['number']).columns

## split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# scale only numeric columns
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[numeric_cols])
X_test_numeric = scaler.transform(X_test[numeric_cols])

# Replace numeric columns in X_train and X_test with scaled values
X_train.loc[:, numeric_cols] = X_train_numeric
X_test.loc[:, numeric_cols] = X_test_numeric

## save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential(
        [
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ]
    )

print(model.summary())

## compile the model

## 1st way
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## 2nd way
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.BinaryAccuracy()
model.compile(optimizer=opt, loss=loss, metrics=[metrics])

## Setup the TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

## setup the early stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


## now this is the time to train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

## load the model
model.save('model.h5')
