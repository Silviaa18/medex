import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import joblib

from sqlalchemy import and_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, TableCategorical, NameType
from sqlalchemy.orm import aliased


def get_entities_for_disease(disease="diabetes"):
    if disease == "diabetes":
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Delta0", "Delta2"]
        # actual database entries
        # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
        # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]

    if disease == "CHD":
        # actual database entries
        # cat_entities = ["alcohol"]
        # num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Jitter_rel", "Jitter_abs"]
    return cat_entities, num_entities


def convert_to_features(df, enc, categorical_columns: list = ['Sex', 'Tobacco smoking']):
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


def load_model():
    model = joblib.load(f'diabetes_prediction_model.pkl')
    scaler = joblib.load(f'diabetes_scaler.pkl')

    encoder = joblib.load(f'diabetes_encoder.pkl')
    return model, encoder, scaler


def get_variable_map(target_disease: str = "diabetes"):
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
            "HbA1c_level": "Glycated haemoglobin (HbA1c)",
            "bmi": "Body mass index (BMI)",
            "blood_glucose_level": "Glucose",
            # "heart_disease": "Diagnoses - ICD10)",
            "gender": "Sex"
        }
    else:
        variable_map = {
            "sbp": "systolic blood pressure automated reading",
            # yearly tobacco use in kg to cigarettes per day
            # "tobacco": "Amount of tobacco currently smoked",
            # yearly alcohol intake(guessing grams/day)? to never, monthly or less, 2 to 4 times a week
            # "alcohol": "Frequency of drinking alcohol",
            "obesity": "Body mass index (BMI)",
            "ldl": "LDL direct",
        }

    return variable_map, smoking_mapping


def train_risk_score_model(target_disease: str = "diabetes", categorical_columns: list = ['Sex', 'Tobacco smoking'],
                           drop_columns: list = ["age", "smoking_history"]):
    # data = pd.read_csv(f'../../examples/{target_disease}_prediction_dataset.csv')
    data = pd.read_csv(f'examples/{target_disease}_prediction_dataset.csv')
    variable_mapping = get_variable_map(target_disease)[0]
    smoking_mapping = get_variable_map(target_disease)[1]
    # year of birth was determined in 2008
    data['Year of birth'] = 2008 - data['age']

    # not current, former, ever,  current, No Info, never to Ex-smoker 2x, Occasionally, Smokes on most or all days,
    # Prefer not to answer, Never smoked
    if target_disease == "diabetes":
        data["Tobacco smoking"] = data["smoking_history"].map(smoking_mapping)

    # Rename variables based on the mapping dictionary
    data = data.rename(columns=variable_mapping)

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
    print(coef_dict)
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


def random_patient(disease="diabetes"):
    # Calculate risk score for a new patient (with the first model)
    if disease == "diabetes":
        patient = {'Sex': 'Male', 'hypertension': 1, 'heart_disease': 1,
                   'Body mass index (BMI)': 30, 'Glycated haemoglobin (HbA1c)': 6.5,
                   'Glucose': 200, 'Year of birth': 1964, 'Tobacco smoking': 'Ex-smoker'}
        # assert get_risk_score(patient)[1] == 0.75190808
    elif disease == "CHD":
        patient = {"systolic blood pressure automated reading": 120, "tobacco": 12, "LDL direct": 5,
                   "Body mass index (BMI)": 32, "alcohol": 97.2, "Year of birth": 1953}
        # risk score should be 0.58063699
    return get_risk_score(patient, disease)


class PredictionService:
    def __init__(self, database_session, filter_service: FilterService):
        self._database_session = database_session
        self._filter_service = filter_service

    @staticmethod
    def get_entities_for_disease(disease="diabetes"):
        if disease == "diabetes":
            cat_entities = ["Gender", "Diabetes"]
            # Actual database entities
            # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
            # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]
            num_entities = ["Delta0", "Delta2"]
        if disease == "CHD":
            cat_entities = []
            # Actual database entities
            # cat_entities = ["alcohol use"]
            # num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
            num_entities = ["Jitter_rel"]
        return cat_entities, num_entities

    def get_risk_score_for_name_id(self, name_id, disease="diabetes") -> dict:

        (cat_entities, num_entities) = PredictionService.get_entities_for_disease(disease)
        if "Diabetes" in cat_entities:
            cat_entities.remove("Diabetes")

        print(cat_entities)

        query = self._database_session.query(
            TableCategorical.name_id,
            TableCategorical.measurement,
            TableCategorical.value.label('Diabetes')
        )

        for i, cat_entity in enumerate(cat_entities):
            tc_alias = aliased(TableCategorical, name=f'TableCategorical{i}')
            query = query.join(tc_alias, and_(
                TableCategorical.name_id == tc_alias.name_id,
                tc_alias.key == cat_entity
            )
                               ).add_columns(tc_alias.value.label(cat_entity))

        for i, num_entity in enumerate(num_entities):
            tn_alias = aliased(TableNumerical, name=f'TableNumerical{i}')
            query = query.join(
                tn_alias,
                and_(
                    tn_alias.name_id == TableCategorical.name_id,
                    tn_alias.key == num_entity
                )
            ).add_columns(tn_alias.value.label(num_entity))

        query = query.filter(TableCategorical.key == 'Diabetes')
        query = query.filter(TableCategorical.name_id == name_id)

        if disease == "CHD":
            drop_columns = ["typea", "famhist", "adiposity", "age"]
        else:
            drop_columns = ["age", 'smoking_history']

        train_risk_score_model(target_disease=disease, drop_columns=drop_columns)
        result = pd.DataFrame(query.all()), random_patient(disease)

        return result


def add_prediction_row(self, entity_key, entity_value, entity_type, default_value, case_id, measurement, date, time):
    query = self._database_session.query(TableCategorical.name_id).distinct()
    cat_name_ids = [row[0] for row in query.all()]

    new_entity = NameType(key=entity_key, synonym=entity_key, description='', unit='', show='', type=entity_type)
    self._database_session.merge(new_entity)

    for name_id in cat_name_ids:
        existing_row = self._database_session.query(TableCategorical).filter_by(name_id=name_id, key=entity_key).first()

        if existing_row is None:
            value = default_value
            prediction_row = TableCategorical(name_id=name_id, key=entity_key, value=value, case_id=case_id,
                                              measurement=measurement, date=date, time=time)
            self._database_session.merge(prediction_row)

    self._database_session.commit()
