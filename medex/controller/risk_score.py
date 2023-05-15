from flask import Blueprint, render_template
from medex.services import data
from medex.controller.helpers import get_prediction_service

risk_controller = Blueprint('risk_controller', __name__)

# example data since we don't know yet how to access the actual database
data = [
    {
        'Disease': 'Diabetes',
        'Score': '20%',
        'Patient': 'John Doe',
        'Birthdate': '08/05/2023'
    }
]


@risk_controller.route('/', methods=['GET'])
def risk_score():
    print("Hello")
    print(get_prediction_service().get_risk_score_for_case_id('5f2b9323c39ee3c861a7b382d205c3d3'))
    print(get_prediction_service().get_risk_score_for_case_id('5890595e16cbebb8866e1842e4bd6ec7', disease="CHD"))

    return render_template('risk_score.html', data=data)

