import pickle as pk
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd
import os
from flask_sqlalchemy import SQLAlchemy

path = os.path.realpath(os.path.dirname(__file__))
path = path.replace("\\", "/")

with open(path + "/rf.pkl", "rb") as model:
    model = pk.load(model)

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:password@host.docker.internal:5432/predictions_log'
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.secret_key = 'password'

    db = SQLAlchemy(app)
    swagger = Swagger(app)
    
    class Predict(db.Model):
        id_prediction = db.Column(db.Integer, primary_key = True)
        type = db.Column(db.String(20), nullable = False)
        sepal_length = db.Column(db.String(10), nullable = False)
        sepal_width = db.Column(db.String(10), nullable = False)
        petal_length = db.Column(db.String(10), nullable = False)
        petal_width = db.Column(db.String(10), nullable = False)
        prediction = db.Column(db.String(10), nullable = False)
        
        def __init__(self, type, sepal_length, sepal_width, petal_length, petal_width, prediction):
            self.type = type
            self.sepal_length = sepal_length
            self.sepal_width = sepal_width
            self.petal_length = petal_length
            self.petal_width = petal_width
            self.prediction = prediction
        
    @app.route('/predict')
    def predict_iris():
        """
        Example endpoint returning a prediction of iris
        ---
        parameters:
            - name: sl
              in: query
              type: number
              required: true
            - name: sw
              in: query
              type: number
              required: true
            - name: pl
              in: query
              type: number
              required: true
            - name: pw
              in: query
              type: number
              required: true
        responses:
            200:
                description: "0: Iris-setosa, 1: Iris-versicolor, 2: Iris-virginica"
        """
        sl = request.args.get('sl')
        sw = request.args.get('sw')
        pl = request.args.get('pl')
        pw = request.args.get('pw')
        type = "Not File"
        prediction = model.predict(np.array([[sl, sw, pl, pw]]))
        
        entry = Predict(type, str(sl), str(sw), str(pl), str(pw), str(prediction[0]))
        db.session.add(entry)
        db.session.commit()
        return str(prediction)

    @app.route('/predict_file', methods=['POST'])
    def predict_iris_file():
        """
        Example endpoint returning a prediction of iris
        ---
        parameters:
            - name: input_file
              in: formData
              type: file
              required: true
        responses:
            200:
                description: "0: Iris-setosa, 1: Iris-versicolor, 2: Iris-virginica"
        """
        input_data = pd.read_csv(request.files.get('input_file'), header=None)
        prediction = model.predict(input_data)
        for i in range(0, input_data.shape[0]):
            temp = []
            type = "File"
            for j in range(0, input_data.shape[1]):
                temp.append(input_data[j][i])
            entry = Predict(type, str(temp[0]), str(temp[1]), str(temp[2]), str(temp[3]), str(prediction[i]))
            db.session.add(entry)
            db.session.commit()
        
        return str(list(prediction))

    return app
