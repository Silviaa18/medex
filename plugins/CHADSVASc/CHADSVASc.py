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
    NEW_KEY_NAME = "CHADSVASc_score"
    ICD10_LABEL_MAPPING = {
        'hypertension': ['I10'],
        'congestive_heart_failure': ['I500'],
        'diabetes': ["E" + str(e) for e in range(100, 149)],
        ## E10 type 1 E11 type 2 E13 other specified E14 other unsp.
        'previous stroke/transient_ischemic_attack/Thrombus': ["I" + str(e) for e
                                                               in range(600, 690)] + ["G458", "G459"] +
                                                              ["I" + str(f) for f in range(800, 810)],
        'atrial_fibrillation': (["I" + str(e) for e in range(480, 483)], False),  # filters this
        'vascular_disease': ["I" + str(e) for e in range(700, 799)]
    }

    @staticmethod
    def entity_type() -> EntityType:
        return EntityType.NUMERICAL

    def calculate(self, df: pd.DataFrame) -> int:
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


def get_plugin_class():
    return CHADSVAScPlugin
