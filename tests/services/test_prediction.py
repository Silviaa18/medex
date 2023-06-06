import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from plugins.plugin2.plugin2 import PredictionService
from tests.fixtures.db_session import db_session
from plugins import plugin2
from medex.database_schema import TableCategorical
<<<<<<< HEAD
from plugins.plugin2.plugin2 import random_patient
from plugins.plugin2.plugin2 import PredictionService
=======
#from medex.services.better_risk_score_model import test_random_patient
#from medex.services.prediction import PredictionService
>>>>>>> c358c88edff6b145f05f1550995ab8fc3113173c


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

        @patch('medex.PredictionService')
        @patch('medex.train_risk_score_model')
        @patch('medex.test_random_patient')
        def test_get_risk_score_for_name_id(self, mock_test_random_patient, mock_train_risk_score_model,
                                            mock_prediction_service):
            # Set up mock objects and data
            mock_query = mock_prediction_service.return_value._database_session.query.return_value
            mock_query.all.return_value = [('name1', 'measurement1', 'value1'), ('name2', 'measurement2', 'value2')]
            mock_test_random_patient.return_value = {'risk_score': 0.5}

            # Create an instance of YourClass
            medex = mock_prediction_service()

            # Call the method under test
            result = medex.get_risk_score_for_name_id('123')

            # Assertions
            self.assertEqual(result,
                             pd.DataFrame([('name1', 'measurement1', 'value1'), ('name2', 'measurement2', 'value2')]))
            mock_prediction_service.get_entities_for_disease.assert_called_once_with('diabetes')
            mock_query.join.assert_called()
            mock_query.filter.assert_called()
            mock_train_risk_score_model.assert_called_once_with(target_disease='diabetes', drop_columns=[])
            mock_test_random_patient.assert_called_once_with('diabetes')


if __name__ == '__main__':
    unittest.main()
