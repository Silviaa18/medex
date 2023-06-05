import pandas as pd
from abc import ABC, abstractmethod
from medex.database_schema import TableNumerical, TableCategorical, NameType
from medex.dto.entity import EntityType
from typing import Union
from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_


class PluginInterface:
    def on_loaded(self):
        pass

    def on_db_ready(self):
        pass

    def on_stopped(self):
        pass

    # ... Expand with further lifecycle methods

class CalculatorInterface(PluginInterface):
    def __init__(self, database_session):
        self._database_session = database_session

    @classmethod
    def get_name(cls) -> str:
        return 'calculate'

    @classmethod
    def required_parameters(cls) -> list[str]:
        return ['df']

    @classmethod
    @abstractmethod
    def get_categorical_keys(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def get_numerical_keys(cls) -> list[str]:
        pass

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> Union[list[TableCategorical], list[TableNumerical]]:
        pass

    @classmethod
    @abstractmethod
    def disease_name(cls):
        pass

    @classmethod
    def entity_type(cls) -> EntityType:
        return EntityType.CATEGORICAL

    @classmethod
    def add_prediction_row(cls, database_session, plugin: 'CalculatorInterface'):
        table = TableCategorical if cls.entity_type() == EntityType.CATEGORICAL else TableNumerical
        prediction_key = f'{cls.disease_name()}_prediction'

        query = cls.build_query(database_session, table, prediction_key)
        df = pd.DataFrame(query.all())
        if df.empty:
            return  # No new patients, early return

        risk_scores = plugin.calculate(df)

        database_session.merge(
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
            database_session.merge(prediction_row)

        database_session.commit()

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
