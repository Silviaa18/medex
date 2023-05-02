from flask import Blueprint, render_template
from medex.services import data

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
    return render_template('risk_score.html', data=data)
