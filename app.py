from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # re-join stemmed tokens
    text = ' '.join(stemmed_tokens)
    
    return text


app = Flask(__name__)

df = pd.read_csv('C:/Users/Admin/Desktop/TEST/tos_main/ans16.1.csv')



v = CountVectorizer()
count = v.fit_transform(df['clauses'])
ans = df['fairness level']


X_train, X_test, Y_train, Y_test = train_test_split(count, ans, test_size=0.2, random_state=48,shuffle=True)

classifier = MultinomialNB()
classifier.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # get the clauses from the HTML form
    clauses = request.form['clauses']
    
    # preprocess the clauses
    processed_clauses = preprocess(clauses)
   

    # make the prediction using the classifier

    prediction = classifier.predict(v.transform([processed_clauses]))[0]

    
    
    # check the prediction value and return appropriate message
    if prediction == [0]:
        return render_template('result.html', prediction='The Clause is Fair')
    elif prediction == [1]:
        return render_template('result.html', prediction='The Clause is Unfair')
    else:
        return render_template('result.html', prediction='Unable to predict fairness level to given Clause.')

if __name__ == '__main__':
    app.run(debug=True)
