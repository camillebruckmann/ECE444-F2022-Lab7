# Import modules and packages
from flask import (
    Flask,
    request,
    render_template,
)
import pickle
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        input_val = request.form
        if input_val != None:
            news = ""
            for key, value in input_val.items():
                if (key == 'text'):
                    news = value
        loaded_model = None
        with open('basic_classifier.pkl','rb') as fid:
            loaded_model = pickle.load(fid)
        
        vectorizer = None
        with open('count_vectorizer.pkl', 'rb') as vd:
            vectorizer = pickle.load(vd)
        prediction = loaded_model.predict(vectorizer.transform([news]))[0]
        if (prediction == 'FAKE'):
            return '1'
        else:
            return '0'

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80, debug=False)
