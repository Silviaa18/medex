from flask import Flask, jsonify, request
from plugins import plugin2
from plugins.plugin2.plugin2 import train_risk_score_model

app = Flask(__name__)


#@app.route("/run_functions", methods=["POST"])
#def run_functions():
 #   train_risk_score_model()
  #  result = test_random_patient()
   # return jsonify(result=result)


