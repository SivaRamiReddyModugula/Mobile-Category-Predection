# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:38:22 2020

@author: Siva Rami Reddy
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={
        'ram':2631,
        'px_height':905,
        'battery_power':1021,
        'px_width':1988,
        'mobile_wt':136,
        'int_memory':53,
        'sc_w':3,
        'talk_time':7,
        'fc':0,
        'sc_h':8
            })

print(r.json())