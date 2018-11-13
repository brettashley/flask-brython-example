import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify


with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, static_url_path="")

@app.route('/model', methods=['GET', 'POST'])
def predict_proba():
    data = request.json
    prediction = model.predict_proba([data['user_input']])
    return jsonify({'not spam': str(round(prediction[0][0], 2)),
                    'spam':str(round(prediction[0][1], 2))})



@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    return jsonify({'probability': random.random()})
