from unittest.mock import MagicMock

import pytest
from tests.fixtures.db_session import db_session

from medex.database_schema import TableCategorical
from medex.services.better_risk_score_model import test_random_patient
from medex.services.prediction import PredictionService


class TestPredictionService:

    def setup_method(self):
        self.database_session = MagicMock()
        self.filter_service = MagicMock()
        self.prediction_service = PredictionService(self.database_session, self.filter_service)

    def test_get_entities_for_disease_diabetes(self):
        cat_entities, num_entities = PredictionService.get_entities_for_disease("diabetes")
        assert cat_entities == ["Gender", "Diabetes"]
        assert num_entities == ["Delta0", "Delta2"]

    def test_get_entities_for_disease_CHD(self):
        cat_entities, num_entities = PredictionService.get_entities_for_disease("CHD")
        assert cat_entities == []
        assert num_entities == ["Jitter_rel"]

    def test_get_risk_score_for_name_id(self):
        name_id = "5f2b9323c39ee3c861a7b382d205c3d3"
        disease = "diabetes"
        cat_entities = ["Gender", "Diabetes"]
        num_entities = ["Delta0", "Delta2"]
        self.database_session.query().filter().all.return_value.return_value = [
            MagicMock(key="Gender", value="Male"),
            MagicMock(key="Diabetes", value="Yes"),
        ]
        train_risk_score_model = MagicMock()
        test_random_patient = MagicMock(return_value="Test Result")
        expected_result = (
            {"Gender": "Male", "Diabetes": "Yes"},
            "Test Result",
        )

        # Call the method being tested
        result = self.prediction_service.get_risk_score_for_name_id(name_id, disease)

        # Perform assertions
        assert result == expected_result
        self.database_session.query.assert_called_once_with(TableCategorical.key, TableCategorical.value)
        self.database_session.query().filter().all.assert_called_once_with()
        train_risk_score_model.assert_called_once_with(target_disease=disease, drop_columns=[])
        test_random_patient.assert_called_once_with(disease)
