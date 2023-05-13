
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


def convert_to_features(df, enc, categorical_columns=['gender', 'smoking_history']):
    onehot = enc.transform(df[categorical_columns])
    df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

    # Concatenate numerical columns with one-hot encoded columns
    columns = list(df.columns)
    for col in categorical_columns:
        columns.remove(col)
    df = pd.concat([df_onehot, df[columns]], axis=1)

    print(df)

    return df


def save_model(model, encoder, scaler):
    joblib.dump(model, 'diabetes_prediction_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')


def load_model(target_disease: str = "diabetes"):
    model = joblib.load(f'{target_disease}_prediction_model.pkl')
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, encoder, scaler


def train_risk_score_model(target_disease: str = "diabetes", categorical_columns=['gender', 'smoking_history']):
    # Load data from CSV file
    data = pd.read_csv(f'examples/{target_disease}_prediction_dataset.csv')

    # Determine categories of categorical columns
    category_names = {}
    for col in categorical_columns:
        category_names[col] = data[col].unique()

    encoder = OneHotEncoder(categories=[category_names[col] for col in categorical_columns])
    scaler = StandardScaler()
    model = LogisticRegression()

    # One-hot encode categorical columns
    encoder.fit(data[categorical_columns])
    data = convert_to_features(data, encoder)
    # Split data into features (X) and target (y)
    x = data.drop([target_disease], axis=1)
    y = data[target_disease]

    # Scale the input features
    x_scaled = scaler.fit_transform(x)

    # Train the logistic regression model using the scaled data
    model.fit(x_scaled, y)

    # Print the model coefficients
    coef_dict = {}
    for coef, feat in zip(model.coef_[0], x.columns):
        coef_dict[feat] = coef
    print(coef_dict)

    # Evaluate model accuracy
    y_pred = model.predict(x_scaled)
    accuracy = accuracy_score(y, y_pred)
    print('Accuracy:', accuracy)

    save_model(model, encoder, scaler)


def get_risk_score(df, model_encoder_scaler=None):
    # Values weren't passed, load from disk
    if model_encoder_scaler is None:
        model, encoder, scaler = load_model()
    else:
        model, encoder, scaler = model_encoder_scaler

    if isinstance(df, dict):
        df = pd.DataFrame(df, index=[0])
        has_disease, risk_score = get_risk_score(df, (model, encoder, scaler))
        return has_disease[0], risk_score[0]

    new_patient_x = convert_to_features(df, encoder)

    # Scale the input features using the same scaler as used in training
    new_patient_x_scaled = scaler.transform(new_patient_x)

    risk_score = model.predict_proba(new_patient_x_scaled)

    print('Risk score:', risk_score[:, 1])

    has_disease = risk_score[:, 1] >= 0.5

    return has_disease, risk_score[:, 1]


def test_random_patient():
    # Calculate risk score for a new patient (with the first model)
    patient = {'gender': 'Male', 'age': 44.0, 'hypertension': 1, 'heart_disease': 1,
               'smoking_history': 'current', 'bmi': 30, 'HbA1c_level': 6.5, 'blood_glucose_level': 200}

    #risk score = 0.7768480562023963
    assert get_risk_score(patient)[0]
    assert get_risk_score(patient)[1] == 0.7768480562023963

    return get_risk_score(patient)





