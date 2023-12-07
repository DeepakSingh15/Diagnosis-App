from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/diabetes')
def diabetesPage():
    return render_template("diabetes.html")

@app.route('/heart')
def heartPage():
    return render_template("heart.html")

@app.route('/doctor')
def doctorPage():
    return render_template("doctor.html")

@app.route('/Predict_Heart',methods=['POST'])
def predictHeart():
    model = pickle.load(open('heart.pkl', 'rb'))
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['cpt']
    data4 = request.form['rbp']
    data5 = request.form['sc']
    data6 = request.form['fbs']
    data7 = request.form['rer']
    data8 = request.form['mhr']
    data9 = request.form['eia']
    data10 = request.form['sdi']
    data11= request.form['slope']
    data12 = request.form['vessel']
    data13= request.form['nfr']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])
    pred = model.predict(arr)
    return render_template('heartResult.html', data=pred)

@app.route('/Predict_Diabetes', methods=['POST'])
def predictDiabetes():
    model = pickle.load(open('diabetes.pkl', 'rb'))
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    return render_template('diabetesResult.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)