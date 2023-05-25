from unittest.mock import MagicMock

import pandas as pd
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

    # not passing
    def test_get_risk_score_for_name_id(self):
        # Create a mock instance of the PredictionService class
        prediction_service_mock = MagicMock()
        # Replace 'PredictionService' with the actual class name in the code being tested

        # Mock the return values and behavior of the methods called within get_risk_score_for_name_id
        prediction_service_mock.get_entities_for_disease.return_value = (
            ["Category1", "Category2"], ["Numeric1", "Numeric2"])
        prediction_service_mock._database_session.query.return_value = MagicMock(
            all=MagicMock(return_value=[
                (1, "Measurement1", 5),
                (1, "Measurement2", 10)
            ])
        )
        prediction_service_mock.test_random_patient.return_value = pd.DataFrame([
            {"RiskScore": 0.75}
        ])

        # Call the method being tested with the mock instance
        result = prediction_service_mock.get_risk_score_for_name_id(1, "diabetes")

        # Perform assertions on the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["RiskScore"], 0.75)

        # Assert that the methods were called with the expected arguments
        prediction_service_mock.get_entities_for_disease.assert_called_once_with("diabetes")
        prediction_service_mock._database_session.query.assert_called_once()
        prediction_service_mock.test_random_patient.assert_called_once_with("diabetes")
