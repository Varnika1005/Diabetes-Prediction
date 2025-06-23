import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Print basic info (optional)
print("First 5 rows of the dataset:")
print(diabetes_dataset.head())

print("\nDataset shape (rows, columns):", diabetes_dataset.shape)

print("\nStatistical description:")
print(diabetes_dataset.describe())

print("\nValue counts for Outcome:")
print(diabetes_dataset['Outcome'].value_counts())

# Grouped mean by Outcome
print("\nMean values grouped by Outcome:")
print(diabetes_dataset.groupby('Outcome').mean())

# 4. Separate features and labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train SVM Classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Evaluate the model
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("\nAccuracy score on training data:", training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy score on test data:", test_data_accuracy)

#  Make a predictive system
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Standardize the new input
std_data = scaler.transform(input_data_as_numpy_array)

# Make prediction
prediction = classifier.predict(std_data)

# Display result
if prediction[0] == 0:
    print("\nThe person is not diabetic.")
else:
    print("\nThe person is diabetic.")
