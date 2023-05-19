from flask import Flask, jsonify, request
from medex.services.better_risk_score_model import train_risk_score_model, test_random_patient

app = Flask(__name__)


@app.route("/run_functions", methods=["POST"])
def run_functions():
    train_risk_score_model()
    result = test_random_patient()
    return jsonify(result=result)


