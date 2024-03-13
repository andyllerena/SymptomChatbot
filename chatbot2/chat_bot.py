# Import necessary libraries
import re
import pandas as pd
import numpy as np
import requests
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# from duckduckgo_search import ddg

all_result = {
    'name':'',
    'age':0,
    'gender':'',
    'symptoms':[]
}

# Load the dataset
df = pd.read_excel('dataset.xlsx')
# Get all unique symptoms
symptoms = set()
for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())

# Get all unique diseases
diseases = set(df['Disease'])

# Define functions

# Function to predict the most relevant symptom based on user input
def predict_symptom(user_input, symptom_list):
    user_input_tokens = user_input.lower().replace("_"," ").split()
    similarity_scores = []
    for symptom in symptom_list:
        symptom_tokens = symptom.lower().replace("_"," ").split()
        # print(symptom_tokens)
        count_vector = np.zeros((2, len(set(user_input_tokens + symptom_tokens))))
        for i, token in enumerate(set(user_input_tokens + symptom_tokens)):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)
            #print(count_vector)
        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]

# Function to predict disease based on symptoms
def predict_disease_from_symptom(symptom_list, df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Symptoms'])
    # print(X)
    # print([', '.join(symptom_list)])
    user_X = vectorizer.transform([', '.join(symptom_list)])

    similarity_scores = cosine_similarity(X, user_X)
    max_score = similarity_scores.max()
    max_indices = similarity_scores.argmax(axis=0)
    diseases = set()
    for i in max_indices:
        if similarity_scores[i] == max_score:
            diseases.add(df.iloc[i]['Disease'])
    if len(diseases) == 0:
        return "No matching diseases found", ""
    elif len(diseases) == 1:
        disease_name = list(diseases)[0]
        return disease_name, get_disease_info(disease_name)
    else:
        return ', '.join(list(diseases)), ""

# Function to get symptoms associated with a disease
def get_symptoms(user_disease, df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])
    similarity_scores = cosine_similarity(X, user_X)
    max_score = similarity_scores.max()
    if max_score < 0.7:
        return False, "No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
        return True, symptoms

# # Function to fetch information about a disease
# def get_disease_info(keywords):
#     results = ddg(keywords, region='wt-wt', safesearch='Off', time='y')
#     return results[0]['body']


def get_disease_info(disease_name):
    api_key = "AIzaSyBWwUoaN2J7C05Z2jp2lbYsuJcmFOFHghc"
    cse_id = "c372f8675b1d04433"
    query = disease_name
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}"

    data = requests.get(url).json()
    try:
        # Fetching the first search result snippet
        info = data['items'][0]['snippet']
        # Optionally, you could return the URL or more details
        # url = data['items'][0]['link']
        return info
    except (KeyError, IndexError):
        return "Information not found or an error occurred."

# preprocess user input
def preprocess_input(input_text):
    # remove punctuation
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))

    # creates an instance of SpellChecker
    spell = SpellChecker()
    # find misspelled words
    misspelled = spell.unknown(input_text.split())
    # replace all misspelled word with correct word
    for word in misspelled:
        corrected_word = spell.correction(word)
        input_text = input_text.replace(word, corrected_word)
    print(input_text)
    # tokenize
    tokens = word_tokenize(input_text)
    
    # stemm the words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


# Main function to interact with the user
def main_chatbot():
    print("Hello, I'm your medical assistant chatbot. Let's get started.")
    name = input("What's your name? ")
    age = input("How old are you? ")
    gender = input("What's your gender? (Male/Female/Other) ")
    while(True):
        symptoms_input = input("Please list your symptoms, separated by commas: ").lower().strip()
        if(symptoms_input != "" and not symptoms_input.isdigit() ):
            break
    # symptoms_list = symptoms_input.split(',')
    symptoms_list = preprocess_input(symptoms_input)
    print(symptoms_list)
    predicted_symptoms = [predict_symptom(symptom.strip(), list(symptoms)) for symptom in symptoms_list]
    predicted_disease, _ = predict_disease_from_symptom(predicted_symptoms, df)
    print(f"Based on the symptoms, the most likely disease is: {predicted_disease}")
    disease_info = get_disease_info(predicted_disease)
    print(f"Here's some information about {predicted_disease}: {disease_info}")



if __name__ == "__main__":

    main_chatbot()

