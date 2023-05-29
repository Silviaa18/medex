import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from medex.services.better_risk_score_model import get_entities_for_disease, load_model, train_risk_score_model, \
    get_risk_score, convert_to_features, save_model

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

import os
from pathlib import Path
from os import getcwd



class MyTestCase(unittest.TestCase):
    def test_get_entities_for_disease(self):
        # Test case for disease = "diabetes"
        cat_entities, num_entities = get_entities_for_disease("diabetes")
        assert cat_entities == ["Gender", "Diabetes"]
        assert num_entities == ["Delta0", "Delta2"]

        # Test case for disease = "CHD"
        cat_entities, num_entities = get_entities_for_disease("CHD")
        assert cat_entities == ["Gender", "Diabetes"]
        assert num_entities == ["Jitter_rel", "Jitter_abs"]


def test_convert_to_features(categorical_columns=["gender", "smoking_history"]):
    # Create a sample DataFrame for testing
    df = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'smoking_history': ['Non-smoker', 'Ex-smoker', 'Current smoker'],
        'age': [30, 35, 40],
        'bmi': [25.0, 27.5, 30.0]
    })

    # Create a sample OneHotEncoder

    category_names = {}
    for col in categorical_columns:
        category_names[col] = df[col].unique()
    enc = OneHotEncoder(categories=[category_names[col] for col in categorical_columns])

    #enc = OneHotEncoder(sparse=False)
    enc.fit(df[['gender', 'smoking_history']])

    # Call the function being tested
    result = convert_to_features(df, enc)
    print(result)

    # Perform assertions on the result
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'gender_Male', 'gender_Female', 'smoking_history_Non-smoker',
                                   'smoking_history_Ex-smoker', 'smoking_history_Current smoker', 'age',
                                   'bmi'}


def test_save_model(folder=""):
    # Create a sample model and encoder
    model = 'Sample Model'
    encoder = 'Sample Encoder'
    scaler = 'Sample Scaler'
    disease = 'diabetes'

    # Call the function being tested
    save_model(model, encoder, scaler, disease)

    # Check if the model and files are saved correctly
    assert Path(folder + 'diabetes_prediction_model.pkl').exists()
    assert Path(folder + 'diabetesencoder.pkl').exists()
    assert Path(folder + 'diabetesscaler.pkl').exists()


def test_load_model(folder=""):
    # Create sample model, encoder, and scaler files
    target_disease = "diabetes"
    model_path = folder + f'{target_disease}_prediction_model.pkl'
    scaler_path = folder + f'{target_disease}scaler.pkl'
    encoder_path = folder + f'{target_disease}encoder.pkl'
    joblib.dump('Sample Model', model_path)
    joblib.dump('Sample Scaler', scaler_path)
    joblib.dump('Sample Encoder', encoder_path)

    # Call the function being tested
    model, encoder, scaler = load_model(target_disease=target_disease)

    # Perform assertions on the returned values
    assert model == 'Sample Model'
    assert encoder == 'Sample Encoder'
    assert scaler == 'Sample Scaler'

    # Call the function again without encoder
    model, encoder, scaler = load_model(target_disease=target_disease, encoder=False)

    # Perform assertions when encoder is not loaded
    assert model == 'Sample Model'
    assert encoder is None
    assert scaler == 'Sample Scaler'


def test_train_risk_score_model_diabetes():
    # Define test data
    target_disease = "diabetes"
    categorical_columns = ['gender', 'smoking_history']
    drop_columns = []

    # Create a dummy CSV file for testing
    test_data = pd.DataFrame({
        'gender': ['Male', 'Female', 'Male'],
        'smoking_history': ['Yes', 'No', 'No'],
        'diabetes': [1, 0, 1]
    })

    # Call the function being tested
    train_risk_score_model(target_disease, categorical_columns, drop_columns)


def test_train_risk_score_model_chd():
    target_disease = "diabetes"
    categorical_columns = ['gender', 'smoking_history']
    drop_columns = []
    current_directory = os.getcwd()
    if current_directory.endswith("tests"):
        parent_directory = os.path.dirname(current_directory)
        os.chdir(parent_directory)
        new_wd = os.getcwd()
    data = pd.read_csv(f'examples/{target_disease}_prediction_dataset.csv')

    # Determine categories of categorical columns
    if target_disease == "diabetes":
        category_names = {}
        for col in categorical_columns:
            category_names[col] = data[col].unique()
        encoder = OneHotEncoder(categories=[category_names[col] for col in categorical_columns])

    scaler = StandardScaler()
    model = LogisticRegression()

    if target_disease == "diabetes":
        # One-hot encode categorical columns
        encoder.fit(data[categorical_columns])
        data = convert_to_features(data, encoder, categorical_columns)

    # Split data into features (X) and target (y)
    y = data[target_disease]
    x = data.drop([target_disease] + drop_columns, axis=1)

    # Scale the input features
    x_scaled = scaler.fit_transform(x)

    # Train the logistic regression model using the scaled data
    model.fit(x_scaled, y)

    # Assert that the model is trained
    assert model is not None
    assert isinstance(model, LogisticRegression)

    # Print the model coefficients
    coef_dict = {}
    for coef, feat in zip(model.coef_[0], x.columns):
        coef_dict[feat] = coef

    # Assert that the coefficients dictionary is not empty
    assert coef_dict != {}

    # Evaluate model accuracy
    y_pred = model.predict(x_scaled)
    accuracy = accuracy_score(y, y_pred)

    # Assert that the accuracy is within an expected range or meets a threshold
    expected_accuracy = 0.8
    assert accuracy >= expected_accuracy

    # Save the model and related objects
    if target_disease == "CHD":
        save_model(model, None, scaler, target_disease)
    else:
        save_model(model, encoder, scaler, target_disease)

    # Assert that the model, encoder, and scaler files are saved
    assert os.path.exists(f'{target_disease}_prediction_model.pkl')
    assert os.path.exists(f'{target_disease}scaler.pkl')
    if target_disease == "diabetes":
        assert os.path.exists(f'{target_disease}encoder.pkl')

    # Clean up the saved files
    os.remove(f'{target_disease}_prediction_model.pkl')
    os.remove(f'{target_disease}scaler.pkl')
    if target_disease == "diabetes":
        os.remove(f'{target_disease}encoder.pkl')


#not working
def test_get_risk_score(expected_has_disease=None, expected_risk_score=None):
    # Mock data
    df = pd.DataFrame({
        'gender': ['Male'],
        'age': [40],
        'smoking_history': ['No'],
        'glucose': [110],
        'bmi': [25.5]
    })

    disease = "diabetes"

    # Mock model, encoder, and scaler
    model = MagicMock()
    encoder = MagicMock()
    scaler = MagicMock()



    # Mock load_model function
    load_model_mock = MagicMock(return_value=(model, encoder, scaler))
    with patch('medex.services.better_risk_score_model.load_model', load_model_mock):
        # Call the function being tested
        result = get_risk_score(df, disease)

        # Perform assertions
        load_model_mock.assert_called_once_with(disease, encoder=True)
        assert result[0] == expected_has_disease
        np.testing.assert_array_equal(result[1], expected_risk_score)

    # Mock convert_to_features function
    convert_to_features_mock = MagicMock(return_value=df)
    with patch('your_module.convert_to_features', convert_to_features_mock):
        # Call the function being tested
        result = get_risk_score(df, disease)

        # Perform assertions
        convert_to_features_mock.assert_called_once_with(df, encoder)
        assert result[0] == expected_has_disease
        np.testing.assert_array_equal(result[1], expected_risk_score)
