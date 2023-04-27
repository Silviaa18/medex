from flask import Flask, render_template
app = Flask(__name__)


data = [
    {
    'Disease': 'Diabetes',
    'Score': '20%',
    'Patient': 'John Doe',
    'Birthdate': '08/05/2023'
    }
]

@app.route('/risk_score')
def risk_score():
    return render_template('risk_score.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)