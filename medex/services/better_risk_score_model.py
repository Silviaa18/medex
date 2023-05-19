
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


def get_entities_for_disease(disease="diabetes"):
    if disease == "diabetes":
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Delta0", "Delta2"]
        # actual database entries
        # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
        # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]

    if disease == "CHD":
        # actual database entries
        # cat_entities = ["alcohol use"]
        # num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Jitter_rel", "Jitter_abs"]
    return cat_entities, num_entities


def convert_to_features(df, enc, categorical_columns: list =['gender', 'smoking_history']):
    assert isinstance(categorical_columns, list)

    onehot = enc.transform(df[categorical_columns])
    df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

    # Concatenate numerical columns with one-hot encoded columns
    columns = list(df.columns)
    for col in categorical_columns:
        columns.remove(col)
    df = pd.concat([df_onehot, df[columns]], axis=1)

    return df


def save_model(model, encoder=None, scaler=None, disease="diabetes"):
    joblib.dump(model, f'{disease}_prediction_model.pkl')
    if encoder:
        joblib.dump(encoder, f'{disease}encoder.pkl')
    joblib.dump(scaler, f'{disease}scaler.pkl')


def load_model(target_disease: str = "diabetes", encoder=True):
    model = joblib.load(f'{target_disease}_prediction_model.pkl')
    scaler = joblib.load(f'{target_disease}scaler.pkl')
    if encoder:
        encoder = joblib.load(f'{target_disease}encoder.pkl')
        return model, encoder, scaler
    else:
        return model, None, scaler


def train_risk_score_model(target_disease: str = "diabetes", categorical_columns: list = ['gender', 'smoking_history'],
                           drop_columns: list = []):
    # Load data from CSV file
    data = pd.read_csv(f'../../examples/{target_disease}_prediction_dataset.csv')
    #data = pd.read_csv(f'examples/{target_disease}_prediction_dataset.csv')

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

    # Print the model coefficients
    coef_dict = {}
    for coef, feat in zip(model.coef_[0], x.columns):
        coef_dict[feat] = coef

    # Evaluate model accuracy
    y_pred = model.predict(x_scaled)
    accuracy = accuracy_score(y, y_pred)
    print('Accuracy:', accuracy)

    if target_disease == "CHD":
        save_model(model, None, scaler, target_disease)
    else:
        save_model(model, encoder, scaler, target_disease)


def get_risk_score(df, disease="diabetes"):
    # Values weren't passed, load from disk
    model, encoder, scaler = load_model(disease, encoder=disease != 'CHD')

    if isinstance(df, dict):
        df = pd.DataFrame(df, index=[0])
        has_disease, risk_score = get_risk_score(df, disease)
        return has_disease[0], risk_score[0]

    if disease != "CHD":
        df = convert_to_features(df, encoder)
    # Scale the input features using the same scaler as used in training
    new_patient_x_scaled = scaler.transform(df)

    risk_score = model.predict_proba(new_patient_x_scaled)

    print('Risk score:', risk_score[:, 1])

    has_disease = risk_score[:, 1] >= 0.5

    return has_disease, risk_score[:, 1]


def test_random_patient(disease="diabetes"):
    # Calculate risk score for a new patient (with the first model)
    if disease == "diabetes":
        patient = {'gender': 'Male', 'age': 44.0, 'hypertension': 1, 'heart_disease': 1,
                   'smoking_history': 'current', 'bmi': 30, 'HbA1c_level': 6.5, 'blood_glucose_level': 200}
        assert get_risk_score(patient)[0]
        assert get_risk_score(patient)[1] == 0.7768480562023963
    elif disease == "CHD":
        patient = {"sbp": 120, "tobacco": 12, "ldl": 5, "obesity": 32, "alcohol": 97.2, "age": 55}
    categorical_columns = get_entities_for_disease(disease)
    return get_risk_score(patient, disease)


train_risk_score_model()
test_random_patient()
