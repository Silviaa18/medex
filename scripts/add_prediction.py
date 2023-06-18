from flask_sqlalchemy.query import Query
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pathlib
import pandas as pd
import joblib
import numpy as np
from medex.database_schema import TableNumerical, TableCategorical, NameType
from typing import Optional
from sqlalchemy.orm import aliased
from sqlalchemy import and_, func, Integer
from medex.services.database import init_db
from abc import ABC, abstractmethod
from medex.dto.entity import EntityType

POSTGRES_USER = 'test'
POSTGRES_PASSWORD = 'test'
POSTGRES_DB = 'example'
POSTGRES_PORT = 5429
POSTGRES_HOST = 'localhost'


def main():
    db = get_db_session()
    #diabetes = DiabetesPredictionPlugin(db)
    #diabetes.on_loaded()
    #diabetes.add_new_rows()
    chd = CHDPredictionPlugin(db)
    chd.on_loaded()
    chd.add_new_rows()


def get_db_session():
    engine = create_engine(
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    init_db(engine, lambda: session)
    return session


class PluginInterface(ABC):
    PLUGIN_NAME = None
    DISEASE_NAME = None
    NUMERICAL_KEYS = None
    CATEGORICAL_KEYS = None
    NEW_KEY_NAME = None
    ICD10_LABEL_MAPPING = None

    def __init__(self, db_session):
        self.table = TableCategorical if self.entity_type() == EntityType.CATEGORICAL else TableNumerical
        self.db_session = db_session

    @staticmethod
    @abstractmethod
    def entity_type() -> EntityType:
        pass

    @abstractmethod
    def calculate(self, df):
        pass

    def on_loaded(self):
        pass

    def add_entity_to_name_type_table(self):
        self.db_session.merge(
            NameType(key=self.NEW_KEY_NAME, synonym=self.NEW_KEY_NAME, description='', unit='', show='',
                     type=self.entity_type().value)
        )

    def icd10_match(self, query):
        # Add conditions to label specific ICD10 values
        for label, icd10_values in self.ICD10_LABEL_MAPPING.items():
            query = query.add_columns((func.sum(
                func.cast(and_(TableCategorical.key == 'Diagnoses - ICD10', TableCategorical.value.in_(icd10_values)),
                          Integer)) > 0).label(label))

        return query

    def _get_db_record_(self, index: tuple, row):
        return self.table(
            name_id=index[0],
            key=self.NEW_KEY_NAME,
            value=row[self.NEW_KEY_NAME],
            case_id=row['case_id'],  # take from input
            measurement=index[1],  # take from input
            date=row['date'],  # take from input
            time=''  # take from input
        )

    def add_new_rows(self):
        query = self.build_query()
        offset = 0
        batch_size = 1000
        columns = self.NUMERICAL_KEYS + (list(self.ICD10_LABEL_MAPPING.keys())
                                         if self.ICD10_LABEL_MAPPING else list()) + self.CATEGORICAL_KEYS

        while True:
            query_batch = query.limit(batch_size).offset(offset)
            df = pd.DataFrame(query_batch.all())
            if df.empty:
                is_first_iteration = offset == 0
                if is_first_iteration:
                    print(f'{self.PLUGIN_NAME}: No new patients found - nothing calculated')
                    return
                break

            self.add_entity_to_name_type_table()

            df.set_index(['name_id', 'measurement'], inplace=True)
            df = df.reindex(sorted(df.columns), axis=1)

            model_df = df[sorted(columns)]
            df[self.NEW_KEY_NAME] = self.calculate(model_df)

            for (name_id, measurement), row in df.iterrows():
                try:
                    prediction_row = self.table(
                        name_id=name_id,
                        key=self.NEW_KEY_NAME,
                        value=row[self.NEW_KEY_NAME],
                        case_id=row['case_id'],  # take from input
                        measurement=measurement,  # take from input
                        date=row['date'],  # take from input
                        time=''  # take from input
                    )
                    self.db_session.add(prediction_row)
                except Exception as e:
                    print(row)
                    print(f"Failed to add record to DB: {str(e)} - skipping")

            self.db_session.commit()
            offset += len(df)

        print(f'{self.DISEASE_NAME}: Added risk scores for {offset} patients')

    def join_on_keys(self, query: Query, current_table, keys: list[str]) -> Query:
        for key in keys:
            alias = aliased(current_table, name='table_' + key)
            query = query.join(
                alias,
                and_(
                    self.table.name_id == alias.name_id,
                    self.table.measurement == alias.measurement,
                    alias.key == key
                )
            ).add_columns(
                # Because of the group by we aggregate but since only one value will be returned it doesn't matter
                func.max(alias.value).label(key)
            )

        return query

    def build_query(self) -> Query:
        query = self.db_session.query(self.table.name_id, self.table.measurement,
                                      func.max(self.table.case_id).label("case_id"),
                                      func.max(self.table.date).label("date")) \
            .group_by(self.table.name_id, self.table.measurement) \
            .having(func.sum(func.cast(self.table.key == self.NEW_KEY_NAME, Integer)) == 0)

        if self.ICD10_LABEL_MAPPING:
            query = self.icd10_match(query)
        query = self.join_on_keys(query, TableCategorical, self.CATEGORICAL_KEYS)
        query = self.join_on_keys(query, TableNumerical, self.NUMERICAL_KEYS)

        return query


class DiabetesPredictionPlugin(PluginInterface):
    PLUGIN_NAME = "DiabetesPredictionPlugin"
    DISEASE_NAME = "diabetes"
    NUMERICAL_KEYS = ["Year of birth", "Glucose", "Body mass index (BMI)",
                      "Glycated haemoglobin (HbA1c)"]
    CATEGORICAL_KEYS = ["Sex", "Tobacco smoking"]
    NEW_KEY_NAME = "Diabetes_prediction"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'heart_disease': ["I"+str(e) for e in range(200, 590)]
    }

    def __init__(self, db):
        super().__init__(db)
        # Only needed if our disease has categorical columns
        self.encoder = None
        self.model = None
        self.scaler = None

    def on_loaded(self):
        target_disease = self.DISEASE_NAME
        self.model = joblib.load(f'{target_disease}_model/prediction_model.pkl')
        self.scaler = joblib.load(f'{target_disease}_model/scaler.pkl')
        self.encoder = joblib.load(f'{target_disease}_model/encoder.pkl')
        print('Diabetes risk score plugin loaded')

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.CATEGORICAL

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        index = df.index
        onehot = self.encoder.transform(df[self.CATEGORICAL_KEYS])
        df_onehot = pd.DataFrame(
            onehot.toarray(),
            columns=self.encoder.get_feature_names_out(self.CATEGORICAL_KEYS),
            index=df.index
        )
        # Concatenate numerical columns with one-hot encoded columns
        columns = list(df.columns)
        for col in self.CATEGORICAL_KEYS:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)

        # Scale numerical columns
        df = self.scaler.transform(df)
        probability_list = self.model.predict_proba(df)[:, 1]
        float_to_string = np.vectorize(lambda x: 'True' if x >= 0.5 else 'False')
        series = pd.Series(float_to_string(probability_list), index=index, name=self.NEW_KEY_NAME)

        return series


class CHDPredictionPlugin(PluginInterface):
    PLUGIN_NAME = "CHDPredictionPlugin"
    DISEASE_NAME = "CHD"
    NUMERICAL_KEYS = ["systolic blood pressure automated reading", "Amount of tobacco currently smoked", "LDL direct",
                      "Year of birth", "Body mass index (BMI)"]
    CATEGORICAL_KEYS = ["Frequency of drinking alcohol"]
    NEW_KEY_NAME = "CHD_prediction"

    def __init__(self, db):
        super().__init__(db)
        # Only needed if our disease has categorical columns
        self.encoder = None
        self.model = None
        self.scaler = None

    def on_loaded(self):
        target_disease = self.DISEASE_NAME
        self.model = joblib.load(f'{target_disease}_model/prediction_model.pkl')
        self.scaler = joblib.load(f'{target_disease}_model/scaler.pkl')
        self.encoder = joblib.load(f'{target_disease}_model/encoder.pkl')
        print('CHD risk score plugin loaded')

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.CATEGORICAL

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        index = df.index
        onehot = self.encoder.transform(df[self.CATEGORICAL_KEYS])
        df_onehot = pd.DataFrame(
            onehot.toarray(),
            columns=self.encoder.get_feature_names_out(self.CATEGORICAL_KEYS),
            index=df.index
        )
        # Concatenate numerical columns with one-hot encoded columns
        columns = list(df.columns)
        for col in self.CATEGORICAL_KEYS:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)

        # Scale numerical columns
        df = self.scaler.transform(df)
        probability_list = self.model.predict_proba(df)[:, 1]
        float_to_string = np.vectorize(lambda x: 'True' if x >= 0.5 else 'False')
        series = pd.Series(float_to_string(probability_list), index=index, name=self.NEW_KEY_NAME)

        return series


main()


# tar cfv script.tar scripts/add_prediction.py scripts/chd_model scripts/diabetes_model