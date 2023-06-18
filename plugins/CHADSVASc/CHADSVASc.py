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
    NUMERICAL_KEYS = ["Year of birth"]
    CATEGORICAL_KEYS = ["Sex"]
    NEW_KEY_NAME = "CHADSVASc"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'congestive_heart_failure': ['I500'],
        'diabetes': ["E" + str(e) for e in range(100, 149)],
        ## E10 type 1 E11 type 2 E13 other specified E14 other unsp.
        'previous stroke/transient_ischemic_attack/Thrombus': ["I" + str(e) for e in range(600, 690)] + ["G458", "G459"],
        #                                                      + ["I" + str(f) for f in range(800, 809)],
        'atrial_fibrillation': (["I" + str(e) for e in range(480, 483)], False),     # filters this
        'vascular_disease': ["I" + str(e) for e in range(700, 799)]
    }

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.NUMERICAL

    def calculate(self, df: pd.DataFrame) -> int:
        score = 0

        for _, row in df.iterrows():
            age = 2008 - row['Year of birth']
            gender = row['Sex']
            hypertension = row['hypertension']
            diabetes = row['diabetes']
            stroke_history = row['previous stroke/transient_ischemic_attack/Thrombus']
            vascular_disease = row['vascular_disease']
            heart_failure = row['congestive_heart_failure']

            score += max(0, age - 65) // 10  # Age ≥ 65: 1 point
            score += 1 if gender == 'female' else 0  # Female gender: 1 point
            score += 1 if hypertension else 0  # Hypertension: 1 point
            score += 1 if diabetes else 0  # Diabetes: 1 point
            score += 2 if stroke_history else 0  # Stroke/TIA history: 2 points
            score += 1 if vascular_disease else 0  # Vascular disease: 1 point
            score += 1 if heart_failure else 0  # Heart failure: 1 point

        return score


def get_plugin_class():
    return CHADSVAScPlugin
