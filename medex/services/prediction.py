from sqlalchemy import select, func, and_, or_
from medex.services.filter import FilterService
from medex.database_schema import TableNumerical, Patient, TableCategorical, TableDate


class PredictionService:
    def __init__(self, database_session, filter_service: FilterService):
        self._database_session = database_session
        self._filter_service = filter_service

    def get_risk_score_for_case_id(self, case_id) -> dict:git

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

        # Convert the query results into a dictionary
        result = {row.key: row.value for row in (qc + qn)}

        return result
