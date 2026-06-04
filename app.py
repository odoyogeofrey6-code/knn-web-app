from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("knn_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    result = model.predict([[hours]])
    return render_template("index.html", result=result[0])

app.run(debug=True)