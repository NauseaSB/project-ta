import numpy as np
import pandas as pd
import catboost as cb
import pickle

from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static')
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template('prediksi.html')


@app.route("/prediksi", methods=["POST"])
def prediksi():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("prediksi.html", prediction_text="{}")


if __name__ == "__main__":
    app.run(debug=True)
