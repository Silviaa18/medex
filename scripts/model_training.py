import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib


def get_entities_for_disease(disease="diabetes"):
    if disease == "diabetes":
        # actual database entries
        cat_entities = ["Sex", "Tobacco smoking"]
        num_entities = ["Year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]

    if disease == "CHD":
        # actual database entries
        cat_entities = ["Frequency of drinking alcohol"]
        num_entities = ["Systolic blood pressure automated reading", "Amount of tobacco currently smoked",
                        "LDL direct", "Year of birth", "Body mass index (BMI)"]
    return cat_entities, num_entities


def convert_to_features(df, enc, categorical_columns: list = ['Sex', 'Tobacco smoking']):

    onehot = enc.transform(df[categorical_columns])
    df_onehot = pd.DataFrame(onehot.toarray(), columns=enc.get_feature_names_out(categorical_columns))

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


def get_variable_map(target_disease: str = "diabetes"):
    alcohol_mapping = {
        'Never': lambda x: x == 0,
        'Monthly or less': lambda x: 0 < x < 0.5,
        '2 to 4 times a month': lambda x: 0.5 <= x <= 1.7,
        'Prefer not to answer': lambda x: 1.7 < x < 2,
        '2 to 3 times a week': lambda x: 2 <= x <= 7,
        '4 or more times a week': lambda x: x > 7
    }
    smoking_mapping = {
        "not current": "Ex-smoker",
        "former": "Ex-smoker",
        "ever": "Occasionally",
        "current": "Smokes on most or all days",
        "No Info": "Prefer not to answer",
        "never": "Never smoked"
    }
    if target_disease == "diabetes":
        variable_map = {
            "bmi": "Body mass index (BMI)",
            "gender": "Sex"
        }
    else:
        variable_map = {
            "sbp": "Systolic blood pressure automated reading",
            "obesity": "Body mass index (BMI)",
            "ldl": "LDL direct",
        }

    return variable_map, smoking_mapping, alcohol_mapping


def train_risk_score_model(target_disease: str = "diabetes", categorical_columns: list = ['Sex', 'Tobacco smoking'],
                           drop_columns: list = ["age", "smoking_history", "HbA1c_level", "blood_glucose_level"]):

    data = pd.read_csv(f'{target_disease}_prediction_dataset.csv')
    variable_mapping = get_variable_map(target_disease)[0]
    smoking_mapping = get_variable_map(target_disease)[1]
    alcohol_mapping = get_variable_map(target_disease)[2]
    # year of birth was determined in 2008
    data['Year of birth'] = 2008 - data['age']

    if target_disease == "diabetes":
        data["Tobacco smoking"] = data["smoking_history"].map(smoking_mapping)
        data["Glycated haemoglobin (HbA1c)"] = (data["HbA1c_level"] - 2.15) * 10.929
        data["Glucose"] = data["blood_glucose_level"] * 0.0555
    if target_disease == "CHD":
        data["Frequency of drinking alcohol"] = data["alcohol"].apply(lambda x: next((key for key, value in
                                                                                      alcohol_mapping.items() if
                                                                                      value(x)), None))
        # tobacco in kg to cigarettes per day
        data['Amount of tobacco currently smoked'] = data['tobacco'] * 0.365

    data = data.rename(columns=variable_mapping)
    data = data.reindex(sorted(data.columns), axis=1)

    category_names = {}
    for col in categorical_columns:
        category_names[col] = data[col].unique()
    encoder = OneHotEncoder(categories=[category_names[col] for col in categorical_columns])

    scaler = StandardScaler()
    model = LogisticRegression()
    encoder.fit(data[categorical_columns])
    data = convert_to_features(data, encoder, categorical_columns)

    y = data[target_disease]
    x = data.drop([target_disease] + drop_columns, axis=1)

    x_scaled = scaler.fit_transform(x)
    model.fit(x_scaled, y)

    coef_dict = {}
    for coef, feat in zip(model.coef_[0], x.columns):
        coef_dict[feat] = coef
    print(coef_dict)
    y_pred = model.predict(x_scaled)
    accuracy = accuracy_score(y, y_pred)
    print('Accuracy:', accuracy)

    save_model(model, encoder, scaler, target_disease)
    print("model trained!")


def get_risk_score(df, disease="diabetes"):
    model, encoder, scaler = load_model(disease)

    if isinstance(df, dict):
        df = pd.DataFrame(df, index=[0])
        has_disease, risk_score = get_risk_score(df, disease)
        return has_disease[0], risk_score[0]

    df = df.reindex(sorted(df.columns), axis=1)

    df = convert_to_features(df, encoder, get_entities_for_disease(disease)[0])
    new_patient_x_scaled = scaler.transform(df)

    risk_score = model.predict_proba(new_patient_x_scaled)

    print('Risk score:', risk_score[:, 1])

    has_disease = risk_score[:, 1] >= 0.5

    return has_disease, risk_score[:, 1]


def random_patient(disease="diabetes"):
    if disease == "diabetes":
        patient = {'Sex': 'Male', 'hypertension': 0, 'heart_disease': 0,
                   'Body mass index (BMI)': 32.8, 'Glycated haemoglobin (HbA1c)': 51.7,
                   'Glucose': 3.96, 'Year of birth': 1949, 'Tobacco smoking': 'Ex-smoker'}
        # risk score should be 0.05582321
    elif disease == "CHD":
        patient = {"Systolic blood pressure automated reading": 129.0, "Amount of tobacco currently smoked": 15.0,
                   "LDL direct": 2.737, "Body mass index (BMI)": 19.8, "Frequency of drinking alcohol":
                       "2 to 3 times a week", "Year of birth": 1965}
        # risk score should be 0.89275387
    return get_risk_score(patient, disease)


train_risk_score_model(
    "CHD",
    categorical_columns=["Frequency of drinking alcohol"],
    drop_columns=["age", "tobacco", "alcohol", "famhist", "typea", "adiposity"]
)
print(random_patient("CHD"))
train_risk_score_model("diabetes")
print(random_patient("diabetes"))
