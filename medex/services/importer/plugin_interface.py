from sqlalchemy.orm import aliased, Query
from sqlalchemy import and_, func, Integer
import pandas as pd
from medex.database_schema import TableNumerical, TableCategorical, NameType
from abc import ABC, abstractmethod
from medex.dto.entity import EntityType


class PluginInterface(ABC):

    PLUGIN_NAME = None
    DISEASE_NAME = None
    NUMERICAL_KEYS = None
    CATEGORICAL_KEYS = None
    NEW_KEY_NAME = None
    ICD10_LABEL_MAPPING = None

    def __init__(self):
        self.table = TableCategorical if self.entity_type() == EntityType.CATEGORICAL else TableNumerical

    @staticmethod
    @abstractmethod
    def entity_type() -> EntityType:
        pass

    @abstractmethod
    def calculate(self, df):
        pass

    def on_loaded(self):
        pass

    def add_entity_to_name_type_table(self, session):
        session.merge(
            NameType(key=self.NEW_KEY_NAME, synonym=self.NEW_KEY_NAME, description='', unit='', show='',
                     type=self.entity_type().value)
        )

    def icd10_match(self, query):
        # Add conditions to label specific ICD10 values
        for label, icd10_values in self.ICD10_LABEL_MAPPING.items():
            query = query.add_columns((func.sum(
                func.cast(and_(self.table.key == 'Diagnoses - ICD10', self.table.value.in_(icd10_values)),
                          Integer)) > 0).label(label))

        return query



    def add_new_rows(self, session):
        query = self.build_query(session)
        offset = 0
        batch_size = 1000
        page = 0
        df = pd.DataFrame(query.limit(batch_size).offset(offset))
        if df.empty:
            print(f'{self.PLUGIN_NAME}: No new patients found - nothing calculated')
            return
        while not df.empty:
            query_batch = query.limit(batch_size).offset(offset)
            df = pd.DataFrame(query_batch.all())
            df.set_index('name_id', inplace=True)
            df = df.reindex(sorted(df.columns), axis=1)
            self.add_entity_to_name_type_table(session)

            values = self.calculate(df)  # calculate gives True/False back
            for name_id, value in zip(df.index, values):

                prediction_row = self.table(
                    name_id=name_id,
                    key=self.NEW_KEY_NAME,
                    value=value,
                    case_id='',  # take from input
                    measurement='',  # take from input
                    date='',  # take from input
                    time=''  # take from input
                )
            session.add(prediction_row)
            session.commit()
            offset += batch_size
            page += 1
        print(f'{self.DISEASE_NAME}: Added risk scores for {page*1000 +len(values)} patients')

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
                # Because of the group by we aggregate but since only one value will be returned it doesn't matter
                func.max(alias.value).label(key)
            )

        return query

    def build_query(self, database_session) -> Query:
        query = database_session.query(self.table.name_id, self.table.measurement) \
            .group_by(self.table.name_id, self.table.measurement) \
            .having(func.sum(func.cast(self.table.key == self.NEW_KEY_NAME, Integer)) == 0)

        if self.ICD10_LABEL_MAPPING:
            query = self.icd10_match(query)
        query = self.join_on_keys(query, TableCategorical, self.CATEGORICAL_KEYS)
        query = self.join_on_keys(query, TableNumerical, self.NUMERICAL_KEYS)

        return query

