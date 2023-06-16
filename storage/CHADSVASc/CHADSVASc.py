import pathlib

import joblib
import pandas as pd
from medex.dto.entity import EntityType
from typing import Optional
import numpy as np
from medex.services.importer.plugin_interface import PluginInterface


class CHADSVAScPlugin(PluginInterface):
    PLUGIN_NAME = "CHADSVAScPlugin"
    DISEASE_NAME = "stroke"
    NUMERICAL_KEYS = ["Year of birth", "Glucose", "Body mass index (BMI)",
                      "Glycated haemoglobin (HbA1c)"]
    CATEGORICAL_KEYS = ["Sex", "Tobacco smoking"]
    NEW_KEY_NAME = "Diabetes_prediction"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'congestive_heart_failure': ['I500'],
        'diabetes': ["E100", "E101", "E102", "E103", "E104", "E105", "E106", "E107", "E108", "E109",    # type 1
                     "E110", "E111", "E112", "E113", "E114", "E115", "E116", "E117", "E118", "E119",    # type 2
                     "E131", "E133", "E135", "E136", "E138", "E139",                                   # other specified
                     "E140", "E141", "E142", "E143", "E144", "E145", "E146", "E147", "E148", "E149",    # other unspeci.
                     ]
        'previous stroke' : ["I600", "I601", "I602", "I604", "I605", "I606", "I607", "I608", "I609"

        ]
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
    return CHADSVAScPlugin
