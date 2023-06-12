import pathlib

import joblib
import pandas as pd
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Optional
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_, func, Integer
from medex.services.importer.plugin_interface import PluginInterface


class CHDPredictionPlugin(PluginInterface):
    def __init__(self):
        # Only needed if our disease has categorical columns
        self.encoder: Optional = None
        self.model = None
        self.scaler = None

    def on_loaded(self):
        target_disease = self.disease_name()
        self.model = joblib.load(f'{target_disease}_model/prediction_model.pkl')
        self.scaler = joblib.load(f'{target_disease}_model/scaler.pkl')
        encoder_file = f'{target_disease}_model/encoder.pkl'

        if pathlib.Path(encoder_file).exists():
            self.encoder = joblib.load(encoder_file)
        print('CHD risk score plugin loaded')

    @classmethod
    def get_categorical_keys(cls) -> list[str]:
        # actual database entries
        return ["alcohol"]
        # return ["Gender", "Diabetes"]

    @classmethod
    def get_numerical_keys(cls) -> list[str]:
        # actual database entries
        return ["sbp", "tobacco", "ldl", "age", "obesity"]
        # return ["Delta0", "Delta2"]

    @classmethod
    def disease_name(cls):
        return 'CHD'

    @classmethod
    def entity_type(cls) -> EntityType:
        return EntityType.CATEGORICAL

    def calculate_risk_score(self, df: pd.DataFrame) -> list[float]:
        if self.encoder is not None:
            onehot = self.encoder.transform(df[self.get_categorical_keys()])
            df_onehot = pd.DataFrame(
                onehot.toarray(),
                columns=self.encoder.get_feature_names_out(self.get_categorical_keys()),
                index=df.index
            )

            # Concatenate numerical columns with one-hot encoded columns
            columns = list(df.columns)
            for col in self.get_categorical_keys():
                columns.remove(col)
            df = pd.concat([df_onehot, df[columns]], axis=1)

        # Scale numerical columns
        df = self.scaler.transform(df)

        return list(self.model.predict_proba(df)[:, 1])

    def on_db_ready(self, session):
        table = TableCategorical if self.entity_type() == EntityType.CATEGORICAL else TableNumerical
        prediction_key = f'{self.disease_name()}_prediction'

        query = self.build_query(session, table, prediction_key)
        df = pd.DataFrame(query.all())

        if df.empty:
            print('CHDPrediction: No new patients found - no risk scores calculated')
            return  # No new patients, early return

        df.set_index('name_id', inplace=True)
        df = df.reindex(sorted(df.columns), axis=1)

        risk_scores = self.calculate_risk_score(df)

        session.merge(
            NameType(key=prediction_key, synonym=prediction_key, description='', unit='', show='', type="String")
        )

        for i, name_id in enumerate(df.index):
            value = 'True' if risk_scores[i] >= 0.5 else 'False'

            prediction_row = table(
                name_id=name_id,
                key=prediction_key,
                value=value,
                case_id=name_id,
                measurement='1',
                date='2011-04-16',  # TODO: Get real datetime
                time='18:50:41'  # TODO: Get real datetime
            )
            session.add(prediction_row)

        session.commit()
        print(f'{self.disease_name()}: Added risk scores for {len(risk_scores)} patients')

    @classmethod
    def join_on_keys(cls, query: Query, base_table, current_table, keys: list[str]) -> Query:
        for key in keys:
            alias = aliased(current_table, name='table_' + key)
            query = query.join(
                alias,
                and_(
                    base_table.name_id == alias.name_id,
                    alias.key == key
                )
            ).add_columns(
                # Because of the group by we aggregate but since only one value will be returned it doesn't matter
                func.max(alias.value).label(key)
            )

        return query

    @classmethod
    def build_query(
            cls,
            database_session,
            table,
            prediction_key: str
    ) -> Query:
        (cat_keys, num_keys) = cls.get_categorical_keys(), cls.get_numerical_keys()

        query = database_session.query(TableCategorical.name_id) \
            .group_by(TableCategorical.name_id) \
            .having(func.sum(func.cast(TableCategorical.key == prediction_key, Integer)) == 0)

        query = cls.join_on_keys(query, table, TableCategorical, cat_keys)
        query = cls.join_on_keys(query, table, TableNumerical, num_keys)

        return query


def get_plugin_class():
    return CHDPredictionPlugin
