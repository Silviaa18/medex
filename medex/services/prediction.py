from sqlalchemy import select, func, and_, or_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, Patient, TableCategorical, TableDate
from medex.services.better_risk_score_model import test_random_patient, get_risk_score, save_model, load_model, train_risk_score_model


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

    def get_risk_score_for_case_id(self, case_id, disease="diabetes") -> dict:

        (cat_entities, num_entities) = PredictionService.get_entities_for_disease(disease)

        print(cat_entities)
        # Execute the query and retrieve the data as a dictionary
        qc = (
            self._database_session.query(TableCategorical.key, TableCategorical.value)
            .filter(TableCategorical.case_id == case_id)
            .filter(TableCategorical.key.in_(cat_entities))
            .all()
        )
        qn = (
            self._database_session.query(TableNumerical.key, TableNumerical.value)
            .filter(TableNumerical.case_id == case_id)
            .filter(TableNumerical.key.in_(num_entities))
            .all()
        )
        if disease == "CHD":
            drop_columns = ["typea", "famhist", "adiposity"]
        else:
            drop_columns = []

        train_risk_score_model(target_disease=disease, drop_columns=drop_columns)

        # Convert the query results into a dictionary
        result = {row.key: row.value for row in (qc + qn)}, test_random_patient(disease)

        return result
