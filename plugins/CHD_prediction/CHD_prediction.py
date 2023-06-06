import pathlib

import joblib
import pandas as pd
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Optional
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_
from medex.services.importer.plugin_interface import PluginInterface

class CHDRiskScorePlugin(PluginInterface):
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
        print('Calculator plugin loaded')

    @classmethod
    def get_categorical_keys(cls) -> list[str]:
        # actual database entries
        # return ["alcohol"]
        return ["Gender", "Diabetes"]

    @classmethod
    def get_numerical_keys(cls) -> list[str]:
        # actual database entries
        # return ["sbp", "tobacco", "ldl", "age", "obesity"]
        return ["Jitter_rel", "Jitter_abs"]

    @classmethod
    def disease_name(cls):
        return 'chd'

    @classmethod
    def entity_type(cls) -> EntityType:
        return EntityType.CATEGORICAL


    def calculate_risk_score(self, df: pd.DataFrame) -> list[float]:
        if self.encoder is not None:
            onehot = self.encoder.transform(df[self.get_categorical_keys()])
            df_onehot = pd.DataFrame(
                onehot.toarray(),
                columns=self.encoder.get_feature_names_out(self.get_categorical_keys())
            )

            # Concatenate numerical columns with one-hot encoded columns
            columns = list(df.columns)
            for col in self.get_categorical_keys():
                columns.remove(col)
            df = pd.concat([df_onehot, df[columns]], axis=1)

        # Scale numerical columns
        df = self.scaler.transform(df)

        return list(self.model.predict(df))

    def on_db_ready(self, session):
        table = TableCategorical if self.entity_type() == EntityType.CATEGORICAL else TableNumerical
        prediction_key = f'{self.disease_name()}_prediction'

        query = self.build_query(session, table, prediction_key)
        df = pd.DataFrame(query.all())
        if df.empty:
            return  # No new patients, early return

        risk_scores = self.calculate_risk_score(df)

        session.merge(
            NameType(key=prediction_key, synonym=prediction_key, description='', unit='', show='', type="String")
        )

        for i, name_id in enumerate(df.index):
            value = 'True' if risk_scores[i] > 0.5 else 'False'

            prediction_row = table.__init__(
                name_id=name_id,
                key=prediction_key,
                value=value,
                case_id=name_id,
                measurement='1',
                date='2011-04-16',  # TODO: Get real datetime
                time='17:50:41'     # TODO: Get real datetime
            )
            session.merge(prediction_row)

        session.commit()

    @classmethod
    def join_on_keys(cls, query: Query, table, keys: list[str]) -> Query:
        for key in keys:
            alias = aliased(table, name=key)
            query = query.join(
                alias,
                and_(
                    table.name_id == alias.name_id,
                    alias.key == key
                )
            ).add_columns(
                alias.value.label(key)
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

        query = database_session.query(table.name_id, table.measurement)
        query = cls.join_on_keys(query, TableCategorical, cat_keys)
        query = cls.join_on_keys(query, TableNumerical, num_keys)

        # Filter out every row that already has prediction_key as an entity
        self_alias = aliased(table, name=prediction_key)
        query.join(
            self_alias,
            and_(
                table.name_id == self_alias.name_id,
                self_alias.key == prediction_key
            ),
            isouter=True,
            full=True
        ).filter(self_alias.value.is_(None))

        return query


#def get_plugin_class():
#    return CHDRiskScorePlugin
