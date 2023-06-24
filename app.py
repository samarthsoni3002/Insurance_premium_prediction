from flask import Flask,render_template,request,app,jsonify,url_for
import pickle 
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns



app = Flask(__name__)

insurance_model = pickle.load(open("./insuranceModel.pkl","rb"))
scaler_model = pickle.load(open("./scalerModel.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")
 

@app.route("/predict",methods=["Post"])
def predict():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(-1,1))
    new_data = scaler_model.transform(np.array(list(data.values())).reshape(-1,1))
    output = insurance_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
        

if __name__=="__main__":
    app.run(debug=True)