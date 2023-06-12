import pathlib

import joblib
import pandas as pd
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Optional
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_, func, Integer
from medex.services.importer.plugin_interface import PluginInterface


class DiabetesPredictionPlugin(PluginInterface):
    PLUGIN_NAME = "DiabetesPredictionPlugin"
    DISEASE_NAME = "diabetes"
    NUMERICAL_KEYS = ['hypertension', 'heart_disease', "Year of birth", "Glucose", "Body mass index (BMI)",
                      "Glycated haemoglobin (HbA1c)"]
    CATEGORICAL_KEYS = ["Sex", "Tobacco smoking", "Diagnoses - ICD10"]
    NEW_KEY_NAME = "Diabetes_prediction"

    def __init__(self):
        super().__init__()
        # Only needed if our disease has categorical columns
        self.encoder: Optional = None
        self.model = None
        self.scaler = None

    def on_loaded(self):
        target_disease = self.DISEASE_NAME
        self.model = joblib.load(f'{target_disease}_model/prediction_model.pkl')
        self.scaler = joblib.load(f'{target_disease}_model/scaler.pkl')
        encoder_file = f'{target_disease}_model/encoder.pkl'

        if pathlib.Path(encoder_file).exists():
            self.encoder = joblib.load(encoder_file)
        print('Diabetes risk score plugin loaded')

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.CATEGORICAL

    def calculate(self, df: pd.DataFrame) -> list[str]:

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
        probability_list = list(self.model.predict_proba(df)[:, 1])
        value_list = ['True' if value >= 0.5 else 'False' for value in probability_list]

        return value_list


def get_plugin_class():
    return DiabetesPredictionPlugin
