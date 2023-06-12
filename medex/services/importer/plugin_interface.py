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

    def add_new_rows(self, session):
        query = self.build_query(session)
        df = pd.DataFrame(query.all())
        # Assuming 'ICD-10' is a column in the DataFrame
        #df['hypertension'] = df['Diagnoses - ICD10'].apply(lambda x: 1 if x == 'I10' else 0)
        #df = df.drop('Diagnoses - ICD10', axis=1)

        if df.empty:
            print(f'{self.PLUGIN_NAME}: No new patients found - nothing calculated')
            return

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
        print(f'{self.DISEASE_NAME}: Added risk scores for {len(values)} patients')

    def join_on_keys(self, query: Query, current_table, keys: list[str]) -> Query:
        for key in keys:
            alias = aliased(current_table, name='table_' + key)
            query = query.join(
                alias,
                and_(
                    self.table.name_id == alias.name_id,
                    alias.key == key
                )
            ).add_columns(
                # Because of the group by we aggregate but since only one value will be returned it doesn't matter
                func.max(alias.value).label(key)
            )

        return query

    def build_query(self, database_session) -> Query:
        query = database_session.query(self.table.name_id) \
            .group_by(self.table.name_id) \
            .having(func.sum(func.cast(self.table.key == self.NEW_KEY_NAME, Integer)) == 0)

        categorical_keys = self.CATEGORICAL_KEYS
        if 'Diagnoses - ICD10' in categorical_keys:
            categorical_keys.remove('Diagnoses - ICD10')
            query = query.add_columns((func.sum(func.cast(and_(self.table.key == 'Diagnoses - ICD10', self.table.value == 'I10'), Integer)) > 0).label('hypertension'))

        query = self.join_on_keys(query, TableCategorical, categorical_keys)
        query = self.join_on_keys(query, TableNumerical, self.NUMERICAL_KEYS)

        return query


