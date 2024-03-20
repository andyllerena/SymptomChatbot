import nltk 
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from collections import defaultdict
import csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import spacy

# Initialize spaCy for preprocessing
nlp = spacy.load("en_core_web_lg")

df = pd.read_excel('C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\raw_data.xlsx')
data = df.fillna(method='ffill')

def process_data(data):
    data_list = []
    data_name = data.replace('^','_').split('_')
    n = 1
    for names in data_name:
        if (n % 2 == 0):
            data_list.append(names)
        n += 1
    return data_list

# Data Cleanup
disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    if (row['Disease'] != "\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(disease)
        count = row['Count of Disease Occurrence']
    if (row['Symptom'] != "\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count

# Preprocess symptoms using spaCy for lemmatization and lowercase
def preprocess_symptoms(symptom_list):
    preprocessed = []
    for symptom in symptom_list:
        doc = nlp(symptom.lower())
        lemmatized = ' '.join(token.lemma_ for token in doc)
        preprocessed.append(lemmatized)
    return preprocessed

# Apply preprocessing to symptoms in the dictionary
for disease, symptoms in disease_symptom_dict.items():
    disease_symptom_dict[disease] = preprocess_symptoms(symptoms)

# Save cleaned data as CSV
with open('cleaned_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for disease, symptoms in disease_symptom_dict.items():
        for symptom in symptoms:
            writer.writerow([disease, symptom, disease_symptom_count[disease]])

# Continue with the existing process to read, encode, and prepare data for training
df = pd.read_csv('cleaned_data.csv', encoding='latin1')
df.columns = ['disease', 'symptom', 'occurrence_count']
df.replace(float('nan'), np.nan, inplace=True)
df.dropna(inplace=True)

n_unique = len(df['symptom'].unique())

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])

# One Hot Encode the Labels
onehot_encoder = OneHotEncoder()
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

cols = np.asarray(df['symptom'].unique())

# Create a new dataframe to save OHE labels
df_ohe = pd.DataFrame(columns = cols)


for i in range(onehot_encoded.shape[0]):
    df_ohe.loc[i] = onehot_encoded[i].toarray()[0]

# Disease Dataframe
df_disease = df['disease']


# Concatenate OHE Labels with the Disease Column
df_concat = pd.concat([df_disease,df_ohe], axis=1)


df_concat.drop_duplicates(keep='first',inplace=True)


cols = df_concat.columns

cols = cols[1:]

# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('disease').sum()
df_concat = df_concat.reset_index()
print (df_concat[:5])

df_concat.to_csv("training_dataset1.csv", index=False)

# One Hot Encoded Features
X = df_concat[cols]

# Labels
y = df_concat['disease']