import numpy as np
import pandas as pd
import catboost as cb
import pickle
import csv
import time
import pickle
from flask import Flask, request, render_template, redirect, url_for
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.experimental import enable_iterative_imputer
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report

from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static')
model = pickle.load(open("model.pkl", "rb"))
data_path = "data/data.csv"


@app.route("/", methods=["GET", "POST"])
def list_data():
    container = []
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        container.append(header)

        for row in reader:
            container.append(row)

    if (request.form.get('page') is not None):
        row = int(request.form.get('page'))
    else:
        row = 100

    sum_data = len(container)
    container = container[:row]

    return render_template("list_data.html", header=header, data=container, row=row, sum_data=sum_data, data_path=data_path)


@app.route("/prediksi", methods=["GET"])
def prediksi():
    return render_template("prediksi.html")


@app.route("/train", methods=["GET"])
def train():
    container = []
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        container.append(header)

        for row in reader:
            container.append(row)

    if (request.form.get('page') is not None):
        row = int(request.form.get('page'))
    else:
        row = 100

    sum_data = len(container)
    container = container[:row]

    return render_template("training.html", sum_data=sum_data, data_path=data_path)


@app.route("/proses_training", methods=["POST"])
def proses_training():
    df = pd.read_csv(data_path)

    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

    no = df[df.RainTomorrow == 0]
    yes = df[df.RainTomorrow == 1]
    yes_oversampled = resample(
        yes, replace=True, n_samples=len(no), random_state=123)
    oversampled = pd.concat([no, yes_oversampled])

    # create a table with data missing
    missing_values = df.isnull().sum()  # missing values

    percent_missing = df.isnull().sum()/df.shape[0]*100  # missing value %

    value = {
        'missing_values ': missing_values,
        'percent_missing %': percent_missing,
        'data type': df.dtypes
    }
    frame = pd.DataFrame(value)

    total = oversampled.isnull().sum().sort_values(ascending=False)
    percent = (oversampled.isnull().sum() /
               oversampled.isnull().count()).sort_values(ascending=False)
    missing = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # Impute categorical var with Mode
    oversampled['Date'] = oversampled['Date'].fillna(
        oversampled['Date'].mode()[0])
    oversampled['Location'] = oversampled['Location'].fillna(
        oversampled['Location'].mode()[0])
    oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(
        oversampled['WindGustDir'].mode()[0])
    oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(
        oversampled['WindDir9am'].mode()[0])
    oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(
        oversampled['WindDir3pm'].mode()[0])

    oversampled.select_dtypes(include=['object']).columns

    # Convert categorical features to continuous features with Label Encoding
    from sklearn.preprocessing import LabelEncoder
    lencoders = {}
    for col in oversampled.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        oversampled[col] = lencoders[col].fit_transform(oversampled[col])

    warnings.filterwarnings("ignore")
    # Multiple Imputation by Chained Equations
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    MiceImputed = oversampled.copy(deep=True)
    mice_imputer = IterativeImputer()
    MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)

    # Detecting outliers with IQR
    Q1 = MiceImputed.quantile(0.25)
    Q3 = MiceImputed.quantile(0.75)
    IQR = Q3 - Q1

    # Removing outliers from the dataset
    MiceImputed = MiceImputed[~(
        (MiceImputed < (Q1 - 1.5 * IQR)) | (MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

    r_scaler = preprocessing.MinMaxScaler()
    r_scaler.fit(MiceImputed)
    modified_data = pd.DataFrame(r_scaler.transform(
        MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)

    X = modified_data.loc[:, modified_data.columns != 'RainTomorrow']
    y = modified_data[['RainTomorrow']]
    selector = SelectKBest(chi2, k=10)
    selector.fit(X, y)
    X_new = selector.transform(X)

    features = MiceImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir',
                            'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
                            'RainToday']]
    target = MiceImputed['RainTomorrow']

    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=12345)

    # Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Catboost
    params_cb = {'iterations': 50,
                 'max_depth': 16}

    model_cb = cb.CatBoostClassifier(**params_cb)

    model_cb, accuracy_cb, roc_auc_cb, coh_kap_cb, tt_cb = run_model(
        model_cb, X_train, y_train, X_test, y_test, verbose=False)

    return render_template("evaluasi.html", accuracy_cb=accuracy_cb, roc_auc_cb=roc_auc_cb, coh_kap_cb=coh_kap_cb, tt_cb=tt_cb, sum_data=len(X_train)+len(X_test))


@app.route("/evaluasi", methods=["GET"])
def evaluasi():
    container = []
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        container.append(header)

        for row in reader:
            container.append(row)

    if (request.form.get('page') is not None):
        row = int(request.form.get('page'))
    else:
        row = 100

    sum_data = len(container)
    container = container[:row]

    return render_template("evaluasi.html", sum_data=sum_data, data_path=data_path)


@app.route("/upload_file", methods=["POST"])
def upload_file():
    file = request.files['file']

    file.save('data/data.csv')

    return redirect(url_for('list_data'))


@app.route("/proses_prediksi", methods=["POST"])
def proses_prediksi():
    # contoh input
    # minTemp = 13.4
    # maxTemp = 22.9
    # rainfall = 0.6
    # evaporation = 1
    # sunshine = 1
    # windGustDir = "W"
    # windGustSpeed = 44
    # windDir9 = "W"
    # windDir3 = "WNW"
    # windSpeed9 = 20
    # windSpeed3 = 24
    # humid9 = 71
    # humid3 = 22
    # pressure9 = 1007.7
    # pressure3 = 1007.1
    # cloud9 = 8
    # cloud3 = 1
    # temp9 = 16.9
    # temp3 = 21.8
    # rainToday = 0
    # location = "Albury"
    # date = "2008-12-01"

    minTemp = float(request.form['minTemp'])
    maxTemp = float(request.form['maxTemp'])
    rainfall = float(request.form['rainfall'])
    evaporation = float(request.form['evaporation'])
    sunshine = float(request.form['sunshine'])
    windGustDir = request.form['windGustDir']
    windGustSpeed = float(request.form['windGustSpeed'])
    windDir9 = request.form['windDir9']
    windDir3 = request.form['windDir3']
    windSpeed9 = float(request.form['windSpeed9'])
    windSpeed3 = float(request.form['windSpeed3'])
    humid9 = float(request.form['humid9'])
    humid3 = float(request.form['humid3'])
    pressure9 = float(request.form['pressure9'])
    pressure3 = float(request.form['pressure3'])
    cloud9 = float(request.form['cloud9'])
    cloud3 = float(request.form['cloud3'])
    temp9 = float(request.form['temp9'])
    temp3 = float(request.form['temp3'])
    rainToday = float(request.form['rainToday'])
    location = request.form['location']
    date = "2008-12-01"

    input_data = pd.DataFrame({
        'Date': [date],
        'Location': [location],
        'MinTemp': [minTemp],
        'MaxTemp': [maxTemp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [windGustDir],
        'WindGustSpeed': [windGustSpeed],
        'WindDir9am': [windDir9],
        'WindDir3pm': [windDir3],
        'WindSpeed9am': [windSpeed9],
        'WindSpeed3pm': [windSpeed3],
        'Humidity9am': [humid9],
        'Humidity3pm': [humid3],
        'Pressure9am': [pressure9],
        'Pressure3pm': [pressure3],
        'Cloud9am': [cloud9],
        'Cloud3pm': [cloud3],
        'Temp9am': [temp9],
        'Temp3pm': [temp3],
        'RainToday': [rainToday],
        'RainTomorrow': [0]
    })

    df = pd.read_csv('data/data.csv')
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

    no = df[df.RainTomorrow == 0]
    yes = df[df.RainTomorrow == 1]
    yes_oversampled = resample(
        yes, replace=True, n_samples=len(no), random_state=123)
    oversampled = pd.concat([no, yes_oversampled])

    oversampled['Date'] = oversampled['Date'].fillna(
        oversampled['Date'].mode()[0])
    oversampled['Location'] = oversampled['Location'].fillna(
        oversampled['Location'].mode()[0])
    oversampled['WindGustDir'] = oversampled['WindGustDir'].fillna(
        oversampled['WindGustDir'].mode()[0])
    oversampled['WindDir9am'] = oversampled['WindDir9am'].fillna(
        oversampled['WindDir9am'].mode()[0])
    oversampled['WindDir3pm'] = oversampled['WindDir3pm'].fillna(
        oversampled['WindDir3pm'].mode()[0])

    lencoders = {}
    for col in oversampled.select_dtypes(include=['object']).columns:
        lencoders[col] = LabelEncoder()
        oversampled[col] = lencoders[col].fit_transform(oversampled[col])

    MiceImputed = oversampled.copy(deep=True)
    mice_imputer = IterativeImputer()
    MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampled)

    Q1 = MiceImputed.quantile(0.25)
    Q3 = MiceImputed.quantile(0.75)
    IQR = Q3 - Q1

    MiceImputed = MiceImputed[~(
        (MiceImputed < (Q1 - 1.5 * IQR)) | (MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

    r_scaler = preprocessing.MinMaxScaler()
    r_scaler.fit(MiceImputed)

    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = lencoders[col].transform(input_data[col])

    # Impute missing values
    input_data = pd.DataFrame(mice_imputer.transform(
        input_data), columns=input_data.columns)

    # Standardize the input data
    input_data = pd.DataFrame(r_scaler.transform(
        input_data), columns=input_data.columns)

    # Make the prediction
    prediction = model.predict(input_data)

    return render_template("hasil_prediksi.html", prediction=prediction, input_data=input_data)


def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0 = time.time()
    if verbose == False:
        model.fit(X_train, y_train, verbose=0)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0

    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test, y_pred, digits=5))

    probs = model.predict_proba(X_test)
    probs = probs[:, 1]
    fper, tper, thresholds = roc_curve(y_test, probs)
    plot_roc_cur(fper, tper)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=model.classes_)
    display.plot(cmap=plt.cm.Blues, xticks_rotation='vertical',
                 values_format='.2f')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.savefig('static/confusion_matrix.png')

    return model, accuracy, roc_auc, coh_kap, time_taken


def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('static/roc.png')


if __name__ == "__main__":
    app.run(debug=True)
