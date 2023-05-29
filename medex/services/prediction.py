from distutils.util import execute

import pandas as pd
import self
from sqlalchemy import select, func, and_, or_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, Patient, TableCategorical, TableDate
from medex.services.better_risk_score_model import test_random_patient, get_risk_score, save_model, load_model, \
    train_risk_score_model
from medex.services.database import get_db_session, get_db_engine
from sqlalchemy.orm import aliased, session
from sqlalchemy.orm import sessionmaker


class PredictionService:
    def __init__(self, database_session, filter_service: FilterService):
        self._database_session = database_session
        self._filter_service = filter_service

    @staticmethod
    def get_entities_for_disease(disease="diabetes"):
        if disease == "diabetes":
            cat_entities = ["Gender", "Diabetes"]
            # Actual database entities
            # cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
            # num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]
            num_entities = ["Delta0", "Delta2"]
        if disease == "CHD":
            cat_entities = []
            # Actual database entities
            # cat_entities = ["alcohol use"]
            # num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
            num_entities = ["Jitter_rel"]
        return cat_entities, num_entities

    def get_risk_score_for_name_id(self, name_id, disease="diabetes") -> dict:

        (cat_entities, num_entities) = PredictionService.get_entities_for_disease(disease)
        if "Diabetes" in cat_entities:
            cat_entities.remove("Diabetes")

        print(cat_entities)

        query = self._database_session.query(
            TableCategorical.name_id,
            TableCategorical.measurement,
            TableCategorical.value.label('Diabetes')
        )

        for i, cat_entity in enumerate(cat_entities):
            tc_alias = aliased(TableCategorical, name=f'TableCategorical{i}')
            query = query.join(tc_alias, and_(
                TableCategorical.name_id == tc_alias.name_id,
                tc_alias.key == cat_entity
            )
                               ).add_columns(tc_alias.value.label(cat_entity))

        for i, num_entity in enumerate(num_entities):
            tn_alias = aliased(TableNumerical, name=f'TableNumerical{i}')
            query = query.join(
                tn_alias,
                and_(
                    tn_alias.name_id == TableCategorical.name_id,
                    tn_alias.key == num_entity
                )
            ).add_columns(tn_alias.value.label(num_entity))

        query = query.filter(TableCategorical.key == 'Diabetes')
        query = query.filter(TableCategorical.name_id == name_id)

        if disease == "CHD":
            drop_columns = ["typea", "famhist", "adiposity"]
        else:
            drop_columns = []

        train_risk_score_model(target_disease=disease, drop_columns=drop_columns)
        result = pd.DataFrame(query.all()), test_random_patient(disease)

        return result


def add_risk_row(self, name_id) -> dict:
    query = self._database_session.query(TableCategorical.name_id).distinct()

    cat_name_ids = [row[0] for row in query.all()]

    print(cat_name_ids)

    for name_id in cat_name_ids:
        risk_row = TableCategorical(name_id=cat_name_ids, entity='risk', value='true')
        query.add(risk_row)
        query.commit()





