from flask_sqlalchemy.query import Query
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
import joblib
import numpy as np
from medex.database_schema import TableNumerical, TableCategorical, NameType
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
    diabetes = DiabetesPredictionPlugin(db)
    diabetes.on_loaded()
    diabetes.add_new_rows()
    chd = CHDPredictionPlugin(db)
    chd.on_loaded()
    chd.add_new_rows()
    chadvasc = CHADSVAScPlugin(db)
    chadvasc.add_new_rows()


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
        icd10_key = 'Diagnoses - ICD10'
        icd10_join = aliased(TableCategorical, name='examination_icd10')
        query = query.join(
            icd10_join,
            and_(
                self.table.name_id == icd10_join.name_id,
                self.table.measurement == icd10_join.measurement,
                icd10_join.key == icd10_key
            ), isouter=True
        )
        # Add conditions to label specific ICD10 values
        for label, icd10_values in self.ICD10_LABEL_MAPPING.items():
            if isinstance(icd10_values, tuple):
                icd10_values, is_optional = icd10_values
            else:
                is_optional = True

            field = (func.sum(
                func.cast(func.coalesce(and_(icd10_join.key == icd10_key, icd10_join.value.in_(icd10_values)), False),
                          Integer)) > 0).label(label)
            query = query.add_columns(field)

            if not is_optional:
                query = query.having(field)

        return query

    def create_row(self, index: tuple, row):
        return self.table(
            name_id=index[0],
            key=self.NEW_KEY_NAME,
            value=row[self.NEW_KEY_NAME],
            case_id=row['case_id'],
            measurement=index[1],
            date=row['date'],
            time=''
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
                    prediction_row = self.create_row((name_id, measurement), row)
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
        columns = list(df.columns)
        for col in self.CATEGORICAL_KEYS:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)
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
        columns = list(df.columns)
        for col in self.CATEGORICAL_KEYS:
            columns.remove(col)
        df = pd.concat([df_onehot, df[columns]], axis=1)
        df = self.scaler.transform(df)
        probability_list = self.model.predict_proba(df)[:, 1]
        float_to_string = np.vectorize(lambda x: 'True' if x >= 0.5 else 'False')
        series = pd.Series(float_to_string(probability_list), index=index, name=self.NEW_KEY_NAME)

        return series


class CHADSVAScPlugin(PluginInterface):
    PLUGIN_NAME = "CHADSVAScPlugin"
    DISEASE_NAME = "stroke"
    NUMERICAL_KEYS = ["Year of birth"]
    CATEGORICAL_KEYS = ["Sex"]
    NEW_KEY_NAME = "CHADSVASc_score"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'congestive_heart_failure': ['I500'],
        'diabetes': ["E" + str(e) for e in range(100, 150)], #130
        ## E10 type 1 E11 type 2 E13 other specified E14 other unsp.
        'previous stroke/transient_ischemic_attack/Thrombus': ["I" + str(e) for e
                                                               in range(600, 700)] + ["G458", "G459"] +  #630
                                                              ["I" + str(f) for f in range(800, 810)],
        'atrial_fibrillation': (["I" + str(e) for e in range(480, 483)] + ["I48"], False),  # filters this
        'vascular_disease': ["I" + str(e) for e in range(700, 799)]  #710
    }

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.NUMERICAL

    def calculate(self, df: pd.DataFrame):
        index = df.index
        score_list = []
        for _, row in df.iterrows():
            score = 0
            age = 2008 - row['Year of birth']
            if age >= 75:
                score += 2
            elif 65 <= age >= 74:
                score += 1
            score += 1 if row['Sex'] == 'Female' else 0
            score += 1 if row['hypertension'] else 0
            score += 1 if row['diabetes'] else 0
            score += 2 if row['previous stroke/transient_ischemic_attack/Thrombus'] else 0
            score += 1 if row['vascular_disease'] else 0
            score += 1 if row['congestive_heart_failure'] else 0
            score_list.append(score)
        series = pd.Series(score_list, index=index, name=self.NEW_KEY_NAME)
        return series


main()
