# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:35:53 2020

@author: Siva Rami Reddy
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    category=''
    
    if output==3:
        category='High range'
    elif output==2:
        category='good range'
    elif output==1:
        category='budject range'
    else:
        category='old model'

    return render_template('index.html', prediction_text='Yours mobile is at {} one'.format(category))


if __name__ == "__main__":
    app.run(debug=True)
