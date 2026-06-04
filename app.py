import os
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)