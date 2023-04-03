from flask import Flask, request, render_template, jsonify, redirect, url_for
from preprocessing import preprocess_corpus
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import numpy as np
import re
import string
import nltk
import math

app = Flask(__name__)


def recommend_by_keyword(query):
    # load dataset
    dataset = pd.read_excel('Dataset Gabungan.xlsx')

    data = np.array(dataset)
    list_abstract = dataset['Abstract']
    lecturer_name = dataset['Dosen']
    lecturer_ID = dataset['ID']

    lecturer_name = np.array(lecturer_name)
    lecturer_ID = np.array(lecturer_ID)
    list_abstract = np.array(list_abstract)

    corpus = preprocess_corpus(dataset['Abstract'].tolist())

    # normalize & TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(text) for text in corpus])
    X_normalized = normalize(X)

    # DBSCAN MODEL
    dbscan = DBSCAN(eps=1, min_samples=2, metric='euclidean')

    # KMEANS MODEL
    # kmeans = KMeans(n_clusters = 51, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    # recommendation code here
    similarity_matrix = 1 / \
        (1 + euclidean_distances(X_normalized, X_normalized))
    labels = dbscan.fit_predict(similarity_matrix)
    # labels = kmeans.fit_predict(similarity_matrix)

    user_input = query.lower()

    # compute similarity to data points
    user_vector = vectorizer.transform([user_input])
    X_normalized_user = normalize(user_vector)
    similarities = 1 / \
        (1 + euclidean_distances(X_normalized_user, X_normalized)[0])

    # find nearest cluster and rank lecturers
    user_cluster_label = labels[np.argmax(similarities)]
    cluster_indices = np.where(labels == user_cluster_label)[0]
    cluster_similarities = similarities[cluster_indices]
    ranked_indices = cluster_indices[np.argsort(cluster_similarities)[::-1]]

    # calculate weights and rank lecturers
    weights = similarities[ranked_indices]
    ranked_lecturers = dataset.iloc[ranked_indices][[
        'Dosen', 'Title', 'Keyword', 'Quota']].to_dict(orient='records')
    result = []
    for i in range(len(ranked_lecturers)):
        if ranked_lecturers[i]['Dosen'] not in [r[0]['Dosen'] for r in result]:
            result.append((ranked_lecturers[i], weights[i]))

    result.sort(key=lambda x: x[1], reverse=True)

    final_result = []
    for lecturer in result:
        if lecturer[0]['Dosen'] not in [r['Dosen'] for r in final_result]:
            final_result.append(lecturer[0])

    recommendations = final_result[:5]

    return recommendations


def get_keyword(recommendations):
    keywords = []
    for recommendation in recommendations:
        keyword = recommendation.get('Keyword')
        if keyword != '-':  # ignore hyphen (-) keywords
            keywords.extend(keyword.split(', '))

    unique_keywords = list(set(keywords))
    return unique_keywords


def recommend_by_academicprofile(inputan):
    # LOAD DATASET ABSTRACT
    dataset = pd.read_excel('Dataset Gabungan.xlsx')

    data = np.array(dataset)
    list_abstract = dataset['Abstract']
    lecturer_name = dataset['Dosen']
    lecturer_ID = dataset['ID']

    lecturer_name = np.array(lecturer_name)
    lecturer_ID = np.array(lecturer_ID)
    list_abstract = np.array(list_abstract)

    # preprocessing data abstract
    corpus = dataset['Abstract'].tolist()
    corpus = preprocess_corpus(dataset['Abstract'].tolist())

    #list_publikasi_clean = np.array(corpus)

    # normalize & TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([' '.join(text) for text in corpus])
    X_normalized = normalize(X)

    # DBSCAN MODEL
    dbscan = DBSCAN(eps=1, min_samples=2, metric='euclidean')

    # KMEANS MODEL
    # kmeans = KMeans(n_clusters = 51, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    similarity_matrix = 1 / (1 + euclidean_distances(X_normalized, X_normalized))
    labels = dbscan.fit_predict(similarity_matrix)
    # labels = kmeans.fit_predict(similarity_matrix)

    # LOAD DATASET SILABUS
    dataset_silabus = pd.read_excel('Dataset Silabus.xlsx')
    list_mata_kuliah = dataset_silabus['Course Name']
    list_silabus_data = dataset_silabus['Syllabus']

    list_mata_kuliah = np.array(list_mata_kuliah)
    list_silabus_data = np.array(list_silabus_data)

    # input yang didapat dari front end
    mata_kuliah_pilihan = []
    nilai_mata_kuliah = []

    for input in inputan:
        print(input)
        mata_kuliah_pilihan.append(input['matkul'])
        nilai_mata_kuliah.append(input['nilai'])

    mata_kuliah_pilihan = np.array(mata_kuliah_pilihan)
    nilai_mata_kuliah = np.array(nilai_mata_kuliah)

    # array untuk menyimpan dokumen dari mata kuliah yang diinputkan
    extra_docs = []

    # memasukkan silabus mata kuliah yang dipilih ke dalam extra document
    for i in range(len(list_mata_kuliah)):
        if list_mata_kuliah[i] in mata_kuliah_pilihan:
            extra_docs.append(list_silabus_data[i])

    extra_docs = np.array(extra_docs)

    user_input = extra_docs

    # remove punctuation and lowercase text
    user_input = [text.translate(str.maketrans(
        '', '', string.punctuation)).lower() for text in user_input]

    # remove stop words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    user_input = [[word for word in text.split() if word not in stop_words]
                  for text in user_input]

    # convert text into numerical vectors using TF-IDF
    X_silabus = vectorizer.transform([' '.join(text) for text in user_input])
    #X_silabus = vectorizer.transform([user_input])

    X_silabus_weighted = []
    for i in range(len(mata_kuliah_pilihan)):
        # temp = X_silabus[i]*credit_input[i]
        temp = nilai_mata_kuliah*X_silabus
        X_silabus_weighted.append(temp)

    X_normalized_silabus = normalize(X_silabus_weighted)

    similarities = 1 / (1 + euclidean_distances(X_normalized_silabus, X_normalized)[0])
    labels_dbscan = dbscan.fit_predict(similarity_matrix)

    # Find nearest cluster and rank lecturers
    user_cluster_label = labels_dbscan[np.argmax(similarities)]
    cluster_indices = np.where(labels_dbscan == user_cluster_label)[0]
    cluster_similarities = similarities[cluster_indices]
    ranked_indices = cluster_indices[np.argsort(cluster_similarities)[::-1]]

    # calculate weights and rank lecturers
    weights = similarities[ranked_indices]
    ranked_lecturers = dataset.iloc[ranked_indices][[
        'Dosen', 'Title', 'Keyword', 'Quota']].to_dict(orient='records')
    result = []
    for i in range(len(ranked_lecturers)):
        if ranked_lecturers[i]['Dosen'] not in [r[0]['Dosen'] for r in result]:
            result.append((ranked_lecturers[i], weights[i]))

    result.sort(key=lambda x: x[1], reverse=True)

    final_result = []
    for lecturer in result:
        if lecturer[0]['Dosen'] not in [r['Dosen'] for r in final_result]:
            final_result.append(lecturer[0])

    recommendations = final_result[:5]

    return recommendations

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/bykeyword', methods=['GET', 'POST'])
def bykeyword():
    if request.method == 'POST':
        # extract the user's query from the form submission
        query = request.form['query']

        # generate a list of recommended theses based on the query
        recommended_theses = recommend_by_keyword(query)
        unique_keywords = get_keyword(recommended_theses)

        # render the recommendations template with the list of recommended theses
        return render_template('result.html', recommendations=recommended_theses, unique_keywords=unique_keywords)
    else:
        # render the bykeyword template
        return render_template('bykeyword.html')


@app.route('/byacademicprofile', methods=['GET', 'POST'])
def byacademicprofile():
    return render_template('byacademicprofile.html')


@app.route('/result_bykeyword', methods=['POST'])
def result():
    # Get the user input from the form
    query = request.form['query']

    # Call the recommender function to get the recommendations
    recommendations = recommend_by_keyword(query)
    unique_keywords = get_keyword(recommendations)

    # Render the result.html template with the recommendations
    return render_template('result.html', recommendations=recommendations, unique_keywords=unique_keywords)


@app.route('/result_byacademicprofile', methods=['POST'])
def result_byacademicprofile():
    try:
        data = request.get_json()
        recommendations = recommend_by_academicprofile(data)
        unique_keywords = get_keyword(recommendations)

        return jsonify({'recommendations': recommendations, 'unique_keywords': unique_keywords})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
