import pathlib

import joblib
import pandas as pd
from medex.dto.entity import EntityType
from typing import Optional
import numpy as np
from medex.services.importer.plugin_interface import PluginInterface


class DiabetesPredictionPlugin(PluginInterface):
    PLUGIN_NAME = "DiabetesPredictionPlugin"
    DISEASE_NAME = "diabetes"
    NUMERICAL_KEYS = ["Year of birth", "Glucose", "Body mass index (BMI)",
                      "Glycated haemoglobin (HbA1c)"]
    CATEGORICAL_KEYS = ["Sex", "Tobacco smoking"]
    NEW_KEY_NAME = "Diabetes_prediction"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'heart_disease': ['I'+str(e) for e in range(200, 509)]
            # I20-25 ischemic HD
            # I34-39 heart valve disorder
            # I42 cardiomyopathy
            # I50 heart failure
    }

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


def get_plugin_class():
    return DiabetesPredictionPlugin
