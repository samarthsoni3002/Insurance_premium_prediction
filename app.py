from flask import Flask,render_template,request,app,jsonify,url_for
import pickle 
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor


app = Flask(__name__)

insurance_model = pickle.load(open("./insuranceModel.pkl","rb"))
scaler_model = pickle.load(open("./scalerModel.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")
 

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler_model.transform(np.array(list(data.values())).reshape(1,-1))
    print(new_data)
    output = insurance_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
            
@app.route("/predict_model",methods=["POST"])
def predict_model():
    data = [float(x) for x in request.form.values()]
    print(data)
    final_input = scaler_model.transform((np.array(data).reshape(1,-1)))
    print(final_input)
    output = insurance_model.predict(final_input)
    print(output)
    return render_template("index.html",prediction_text = "The predicted price is {}".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)