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
        icd10_key = 'Diagnoses - ICD10'
        icd10_join = aliased(TableCategorical, name='examination_icd10')
        query = query.join(
            icd10_join,
            and_(
                self.table.name_id == icd10_join.name_id,
                self.table.measurement == icd10_join.measurement,
                icd10_join.key == icd10_key
            ), isouter=True
        )
        # Add conditions to label specific ICD10 values
        for label, icd10_values in self.ICD10_LABEL_MAPPING.items():
            if isinstance(icd10_values, tuple):
                icd10_values, is_optional = icd10_values
            else:
                is_optional = True

            field = (func.sum(func.cast(func.coalesce(and_(icd10_join.key == icd10_key, icd10_join.value.in_(icd10_values)), False), Integer)) > 0).label(label)
            query = query.add_columns(field)

            if not is_optional:
                query = query.having(field)

        return query

    def create_prediction_row(self, index: tuple, row):
        return self.table(
            name_id=index[0],
            key=self.NEW_KEY_NAME,
            value=row[self.NEW_KEY_NAME],
            case_id=row['case_id'],
            measurement=index[1],
            date=row['date'],
            time=''
        )

    def add_new_rows(self, session):
        query = self.build_query(session)
        offset = 0
        batch_size = 1000
        columns = self.NUMERICAL_KEYS + (list(self.ICD10_LABEL_MAPPING.keys()) if self.ICD10_LABEL_MAPPING else list()) \
                  + self.CATEGORICAL_KEYS

        while True:
            query_batch = query.limit(batch_size).offset(offset)
            df = pd.DataFrame(query_batch.all())
            if df.empty:
                is_first_iteration = offset == 0
                if is_first_iteration:
                    print(f'{self.PLUGIN_NAME}: No new patients found - nothing calculated')
                    return
                break

            self.add_entity_to_name_type_table(session)

            df.set_index(['name_id', 'measurement'], inplace=True)
            df = df.reindex(sorted(df.columns), axis=1)

            model_df = df[sorted(columns)]
            df[self.NEW_KEY_NAME] = self.calculate(model_df)

            for (name_id, measurement), row in df.iterrows():
                try:
                    prediction_row = self.create_prediction_row((name_id, measurement), row)
                    session.add(prediction_row)
                except Exception as e:
                    print(row)
                    print(f"Failed to add record to DB: {str(e)} - skipping")

            session.commit()
            offset += len(df)

        print(f'{self.DISEASE_NAME}: Added risk scores for {offset} patients')

    def join_on_keys(self, query: Query, current_table, keys: list[str]) -> Query:
        for key in keys:
            alias = aliased(current_table, name='table_' + key)
            query = query.join(
                alias,
                and_(
                    self.table.name_id == alias.name_id,
                    self.table.measurement == alias.measurement,
                    alias.key == key
                ),
            ).add_columns(
                # Because of the group by we aggregate but since only one value will be returned it doesn't matter
                func.max(alias.value).label(key)
            )

        return query

    def build_query(self, database_session) -> Query:
        query = database_session.query(self.table.name_id, self.table.measurement,
                                       func.max(self.table.case_id).label("case_id"),
                                       func.max(self.table.date).label("date")) \
            .group_by(self.table.name_id, self.table.measurement) \
            .having(func.sum(func.cast(self.table.key == self.NEW_KEY_NAME, Integer)) == 0)

        if self.ICD10_LABEL_MAPPING:
            query = self.icd10_match(query)
        query = self.join_on_keys(query, TableCategorical, self.CATEGORICAL_KEYS)
        query = self.join_on_keys(query, TableNumerical, self.NUMERICAL_KEYS)

        return query
