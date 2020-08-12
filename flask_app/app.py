from flask import Flask, render_template, request
# from sklearn.externals import joblib
#from sklearn.ensemble import RandomForestClassifier
import pickle
import sklearn.externals
import pandas as pd
import numpy as np

app = Flask(__name__)
ml_model = pickle.load(open('rm.pkl','rb'))

# mul_reg = open("rm.pkl", "rb")
# ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('NewYork'))
        try:
            time_in_hospital = float(request.form['time_in_hospital'])
            num_lab_procedures = float(request.form['num_lab_procedures'])
            age = float(request.form['age'])
            num_medications = float(request.form['num_medications'])
            num_procedures = float(request.form['num_procedures'])
            number_diagnoses = float(request.form['number_diagnoses'])
            pred_args = [time_in_hospital,num_lab_procedures,age,num_medications,num_procedures,number_diagnoses]

            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)

            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)

        except ValueError:
            return "Please check if the values are entered correctly"

    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(debug=True)
