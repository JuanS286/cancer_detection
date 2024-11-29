# Breast Cancer Prediction with Artificial Neural Networks

This repository contains the implementation of a Breast Cancer Prediction system using an Artificial Neural Network (ANN). The project involves training an ANN on the Breast Cancer dataset and deploying the model as an interactive web application using Streamlit.

## Project Structure


## Features

### Training and Evaluation
- The ANN is trained on the Breast Cancer dataset from `sklearn`.
- Feature selection is applied using `SelectKBest`.
- The dataset is preprocessed, scaled, and split into training and testing sets.
- The model is evaluated on metrics such as:
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - AUC

### Deployment
- The trained model is deployed as a Streamlit app.
- Users can input feature values, and the app predicts whether the tumor is **Malignant** or **Benign**.
- Predictions include confidence levels.

## Jupyter Notebook (`assingment4.ipynb`)

The notebook includes:

1. **Data Preparation**:
   - Loading the Breast Cancer dataset.
   - Feature scaling and selection using `SelectKBest`.

2. **Model Training**:
   - Building an ANN using TensorFlow/Keras.
   - Hyperparameter tuning using GridSearchCV.
   - Saving the trained model and scaler for deployment.

3. **Evaluation**:
   - Metrics: Accuracy, F1 Score, Precision, Recall, AUC.
   - Confusion matrix visualization.

## Streamlit App (`app.py`)

The app provides an intuitive interface for users to input values for the selected features and receive predictions.

### Features:
1. **Dynamic Input Fields**:
   - Input fields for 15 selected features, each constrained to its valid range.

2. **Model Prediction**:
   - Processes the input through the trained model.
   - Displays the predicted class (**Malignant** or **Benign**) and confidence level.

3. **Preprocessing**:
   - The scaler used during training ensures consistent preprocessing in the app.

