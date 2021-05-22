# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:57:13 2020

@author: Dindi Divya
"""

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
app = Flask(__name__)
model = load_model("project.h5")
@app.route('/')
def index():
    return render_template('base.html')
@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        with graph.as_default():
            preds = model.predict_classes(x)
            print("prediction",preds)
        index = ['iceberg','ship']
        if(index[preds[0][0]]=='iceberg'):
            text = "Given image consists iceberg"
        else:
            text = "Given image doesn't contains iceberg"
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)