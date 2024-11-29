# Breast Cancer Prediction with Artificial Neural Networks

## App link: https://juans286-cancer-detection-app-lg9q9g.streamlit.app/

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
   - I set SelectKBest to select 15 out of the 30 features. The final features are: ['mean radius', 'mean perimeter',
     'mean area', 'mean compactness', 'mean concavity', 'mean concave points', 'radius error','perimeter error', 'area error',
     'worst radius', 'worst perimeter', 'worst area', 'worst compactness', 'worst concavity', 'worst concave points']

2. **Model Training**:
   - Hyperparameter tuning using GridSearchCV. As a result the best parameters are:
     {'alpha': 0.0001,
     'hidden_layer_sizes': (64, 32),
     'learning_rate': 'constant',
     'max_iter': 2000,
     'solver': 'adam'}
   - Building an ANN using TensorFlow/Keras.
   - Saving the trained model and scaler for deployment.

4. **Evaluation**:
   - Metrics: Accuracy, AUC.

The result of the model evaluation is presented with the folowing plots:

![image](https://github.com/user-attachments/assets/246af629-226e-4348-8722-f5a0952e1802)

![image](https://github.com/user-attachments/assets/d79f9a5a-3ac6-4e8e-adba-1e1b220682dd)

![image](https://github.com/user-attachments/assets/ae495483-cd1c-4359-b5da-e8482ebae076)

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

