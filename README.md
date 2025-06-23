# 🧠 Diabetes Prediction using Machine Learning (SVM)

This project is a machine learning-based system to predict whether a person is diabetic or not using the **Pima Indians Diabetes Dataset**. The model is built using Python and trained using an SVM (Support Vector Machine) classifier.

## 📌 Features

- Loads and explores the diabetes dataset
- Standardizes input features using `StandardScaler`
- Splits data into training and test sets
- Trains a Support Vector Machine classifier with a linear kernel
- Evaluates performance using accuracy score
- Predicts diabetes outcome for new input data

## 🧪 Dataset Information

The dataset includes the following medical parameters:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## 🎯 Model Performance

- **Training Accuracy**: 78.66%
- **Testing Accuracy**: 77.27%

## 🧮 Example Prediction

```python
input_data = (5,166,72,19,175,25.8,0.587,51)
# Output: This person is Diabetic
