import pytest
from copy import deepcopy
from flask import Flask

from medex.controller.filter import filter_controller
from tests.mocks.filter_service import FilterServiceMock, DEFAULT_FILTER_STATUS


@pytest.fixture
def filter_service_mock():
    yield FilterServiceMock()


stored_filter_status = None


@pytest.fixture
def helper_mock(filter_service_mock, mocker):
    def _store_filter_status(x):
        global stored_filter_status
        stored_filter_status = x.dict()

    global stored_filter_status
    stored_filter_status = deepcopy(DEFAULT_FILTER_STATUS)
    mocker.patch(
        'medex.controller.filter.get_filter_service',
        return_value=filter_service_mock
    )
    mocker.patch(
        'medex.controller.filter.store_filter_status_in_session',
        new=_store_filter_status
    )


@pytest.fixture
def test_client():
    app = Flask(__name__)
    app.register_blueprint(filter_controller)
    yield app.test_client()


def test_delete_one_filter(helper_mock, test_client):
    rv = test_client.delete(
        '/delete',
        json={'entity': 'diabetes'}
    )
    assert rv.status == '200 OK'
    assert stored_filter_status == {
        'filters': {
            'temperature': {'from_value': 39.0, 'to_value': 43.0, 'min': 30.0, 'max': 43.0}
        }
    }


def test_delete_all_filters(helper_mock, test_client):
    rv = test_client.delete('/delete_all')
    assert rv.status == '200 OK'
    assert stored_filter_status == {'filters': {}}


def test_add_categorical_filter(helper_mock, test_client):
    rv = test_client.post(
        '/add_categorical',
        json={'entity': 'ebola', 'categories': ['ja', 'vielleicht']},
    )
    assert rv.status == '200 OK'
    assert stored_filter_status == {
        'filters': {
            **DEFAULT_FILTER_STATUS['filters'],
            'ebola': {'categories': ['ja', 'vielleicht']},
        }
    }


def test_add_numerical_filter(helper_mock, test_client):
    rv = test_client.post(
        '/add_numerical',
        json={'entity': 'Größe cm', 'from_value': 170, 'to_value': 180, 'min': 30, 'max': 300},
    )
    assert rv.status == '200 OK'
    assert stored_filter_status == {
        'filters': {
            **DEFAULT_FILTER_STATUS['filters'],
            'Größe cm': {'from_value': 170, 'to_value': 180, 'min': 30, 'max': 300},
        }
    }