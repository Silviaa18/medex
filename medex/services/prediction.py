from sqlalchemy import select, func, and_, or_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, Patient, TableCategorical, TableDate
from medex.services.better_risk_score_model import test_random_patient, get_risk_score, save_model, load_model, train_risk_score_model

class PredictionService:
    def __init__(self, database_session, filter_service: FilterService):
        self._database_session = database_session
        self._filter_service = filter_service

    def get_risk_score_for_case_id(self, case_id) -> dict:

        # Execute the query and retrieve the data as a dictionary
        qc = (
            self._database_session.query(TableCategorical.key, TableCategorical.value)
            .filter(TableCategorical.case_id == case_id)
            .filter(or_(TableCategorical.key == "Diabetes", TableCategorical.key == "Gender"))
            .all()
        )
        qn = (
            self._database_session.query(TableNumerical.key, TableNumerical.value)
            .filter(TableNumerical.case_id == case_id)
            .filter(or_(TableNumerical.key == "Delta0", TableNumerical.key == "Delta2"))
            .all()
        )
        train_risk_score_model()
        # Convert the query results into a dictionary
        result = {row.key: row.value for row in (qc + qn)}, test_random_patient()

        return result
