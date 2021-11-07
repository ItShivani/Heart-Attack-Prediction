import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('heart_attack_pred_mlp_model.pickle', 'rb'))

@app.route('/',methods=['GET'])
def home():
	if request.method=='GET':
		return render_template('predictpage.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = " High risk of Heart Attack"
    else:
        res_val = "Low risk of Heart Attack"
        

    return render_template('predictpage.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
