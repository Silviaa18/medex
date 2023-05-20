import pandas as pd
from sqlalchemy import select, func, and_, or_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, Patient, TableCategorical, TableDate
from medex.services.better_risk_score_model import test_random_patient, get_risk_score, save_model, load_model, train_risk_score_model
from medex.services.database import get_db_session, get_db_engine
from sqlalchemy.orm import aliased


class PredictionService:
    def __init__(self, database_session, filter_service: FilterService):
        self._database_session = database_session
        self._filter_service = filter_service

    @staticmethod
    def get_entities_for_disease(disease="diabetes"):
        if disease == "diabetes":
            cat_entities = ["Gender", "Diabetes"]
            #cat_entities = ["Diagnoses - ICD10", "Sex", "Tobacco smoking"]
            #num_entities = ["year of birth", "Glucose", "Body mass index (BMI)", "Glycated haemoglobin (HbA1c)"]
            num_entities = ["Delta0", "Delta2"]
        if disease == "CHD":
            cat_entities = []
            #cat_entities = ["alcohol use"]
            #num_entities = ["sbp", "tobacco", "ldl", "age", "obesity"]
            num_entities = ["Jitter_rel"]
        return cat_entities, num_entities

    def get_risk_score_for_name_id(self, name_id, disease="diabetes") -> dict:

        (cat_entities, num_entities) = PredictionService.get_entities_for_disease(disease)

        print(cat_entities)
        # Execute the query and retrieve the data as a dictionary
        #qc = (
        #    self._database_session.query(TableCategorical.key, TableCategorical.value)
        #    .filter(TableCategorical.name_id == name_id)
        #    .filter(TableCategorical.key.in_(cat_entities))
        #    .all()
        #)
        #qn = (
        #    self._database_session.query(TableNumerical.key, TableNumerical.value)
        #    .filter(TableNumerical.name_id == name_id)
        #    .filter(TableNumerical.key.in_(num_entities))
        #    .all()
        #)
        tc2 = aliased(TableCategorical, name='TableCategorical2')
        tc3 = aliased(TableCategorical, name='TableCategorical3')
        tn2 = aliased(TableNumerical, name='TableNumerical2')
        tn3 = aliased(TableNumerical, name='TableNumerical3')
        query = self._database_session.query(
            TableCategorical.name_id,
            TableCategorical.measurement,
            TableCategorical.value.label('Diabetes'),
            tc2.value.label('Gender'),
            tn3.value.label('Delta0'),
            tn2.value.label('Delta2')
        ).join(
            tc2,
            and_(TableCategorical.name_id == tc2.name_id, tc2.key == 'Gender')
        ).join(
            tn2,
            and_(tn2.name_id == TableCategorical.name_id, tn2.key == 'Delta2')
        ).join(
            tn3,
            and_(TableCategorical.name_id == tn3.name_id, tn3.key == 'Delta0')
        ).filter(
            TableCategorical.key == 'Diabetes'
        ).filter(TableCategorical.name_id == name_id)
        #sql_query = query.statement.compile(engine, compile_kwargs={"literal_binds": True}).string

        #print(sql_query)
        print(query)
        if disease == "CHD":
            drop_columns = ["typea", "famhist", "adiposity"]
        else:
            drop_columns = []

        train_risk_score_model(target_disease=disease, drop_columns=drop_columns)
        # Convert the query results into a dictionary
        result = pd.DataFrame(query.all()), test_random_patient(disease)

        return result


#session = get_db_session()
#x = PredictionService(None, FilterService)
#print(PredictionService.get_risk_score_for_name_id(x, '5f2b9323c39ee3c861a7b382d205c3d3'))
