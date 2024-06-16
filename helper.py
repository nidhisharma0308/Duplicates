import pickle
import numpy as np
import pickle

# Load the count vectorizer
cv = pickle.load(open('cv.pkl','rb'))

def split(text):
    return len(text.split())

def common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))

def query_point_creator(q1, q2):
    input_query = []

    # Fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    input_query.append(common_words(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2) / total_words(q1, q2), 2))

    # BOW feature for q1
    q1_bow = cv.transform([q1]).toarray()

    # BOW feature for q2
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 7), q1_bow, q2_bow))
