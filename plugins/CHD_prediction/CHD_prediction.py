import pathlib

import joblib
import pandas as pd
import numpy as np
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Optional
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_, func, Integer
from medex.services.importer.plugin_interface import PluginInterface


class CHDPredictionPlugin(PluginInterface):
    PLUGIN_NAME = "CHDPredictionPlugin"
    DISEASE_NAME = "CHD"
    NUMERICAL_KEYS = ["Systolic blood pressure automated reading", "Amount of tobacco currently smoked", "LDL direct",
                      "Year of birth", "Body mass index (BMI)"]
    CATEGORICAL_KEYS = ["Frequency of drinking alcohol"]
    NEW_KEY_NAME = "CHD_prediction"

    def __init__(self):
        super().__init__()
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

#    def calculate_risk_score(self, df: pd.DataFrame) -> list[float]:
#        if self.encoder is not None:
#            onehot = self.encoder.transform(df[self.get_categorical_keys()])
#            df_onehot = pd.DataFrame(
#                onehot.toarray(),
#                columns=self.encoder.get_feature_names_out(self.get_categorical_keys()),
#                index=df.index
#            )
#
#            # Concatenate numerical columns with one-hot encoded columns
#            columns = list(df.columns)
#            for col in self.get_categorical_keys():
#                columns.remove(col)
#            df = pd.concat([df_onehot, df[columns]], axis=1)
#
#        # Scale numerical columns
#        df = self.scaler.transform(df)
#
#        return list(self.model.predict_proba(df)[:, 1])
#
#    def on_db_ready(self, session):
#        table = TableCategorical if self.entity_type() == EntityType.CATEGORICAL else TableNumerical
#        prediction_key = f'{self.disease_name()}_prediction'
#
#        query = self.build_query(session, table, prediction_key)
#        df = pd.DataFrame(query.all())
#
#        if df.empty:
#            print('CHDPrediction: No new patients found - no risk scores calculated')
#            return  # No new patients, early return
#
#        df.set_index('name_id', inplace=True)
#        df = df.reindex(sorted(df.columns), axis=1)
#
#        risk_scores = self.calculate_risk_score(df)
#
#        session.merge(
#            NameType(key=prediction_key, synonym=prediction_key, description='', unit='', show='', type="String")
#        )
#
#        for i, name_id in enumerate(df.index):
#            value = 'True' if risk_scores[i] >= 0.5 else 'False'
#
#            prediction_row = table(
#                name_id=name_id,
#                key=prediction_key,
#                value=value,
#                case_id=name_id,
#                measurement='1',
#                date='2011-04-16',  # TODO: Get real datetime
#                time='18:50:41'  # TODO: Get real datetime
#            )
#            session.add(prediction_row)
#
#        session.commit()
#        print(f'{self.disease_name()}: Added risk scores for {len(risk_scores)} patients')
#

def get_plugin_class():
    return CHDPredictionPlugin
