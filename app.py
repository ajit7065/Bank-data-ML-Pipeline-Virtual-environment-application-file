from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('Bank_With_Pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = request.form['age']
    job = request.form['job']
    marital = request.form['marital']
    education = request.form['education']
    default = request.form['default']
    balance = request.form['balance']
    housing = request.form['housing']
    loan = request.form['loan']
    contact = request.form['contact']
    day = request.form['day']
    month = request.form['month']
    duration = request.form['duration']
    campaign = request.form['campaign']
    pdays = request.form['pdays']
    previous = request.form['previous']
    poutcome = request.form['poutcome']

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Determine the prediction result
    result = "Yes" if prediction[0] == 1 else "No"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
