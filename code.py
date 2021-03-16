import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,sigmoid_kernel
import json
import bs4 as bs
import urllib.request
import pickle
import requests
data = pd.read_csv(r'final_all')
movie_series=pd.Series(index=data['movie_title'],data=data['movie_title'].index)



app = Flask(__name__)
@app.route("/")
@app.route("/home")

def home():
    return render_template('home.html')


def similarity():
    tfidf= TfidfVectorizer(ngram_range=(1,3),min_df=3)
    count_matrix = tfidf.fit_transform(data['comb']).toarray()
    # creating a similarity score matrix
    sig = sigmoid_kernel(count_matrix,count_matrix)
    return data, sig

@app.route("/recommend",methods=["POST"])

def recommend(x):
    a,b=similarity()
    sig=b
    lis = []
    lis2 = []
    index = movie_series[x]
    values = sig[index]
    for i, j in enumerate(values):
        lis.append((i, j))
    lis = sorted(lis, key=lambda x: x[1], reverse=True)
    lis = lis[:5]
    for item in lis:
        lis2.append(item[0])
    out=data.iloc[lis2, 0:1]
    return render_template('recommend.html',out=out)




if __name__ == '__main__':
    app.run(debug=True)




