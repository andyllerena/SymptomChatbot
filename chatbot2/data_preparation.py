# Import necessary libraries
import nltk 
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn import preprocessing
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Initialize a Porter Stemmer for word stemming
stemmer = PorterStemmer()

# Tokenize function: Splits a sentence into individual words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stemming function: Reduces a word to its base or root form
def stem(word):
    return stemmer.stem(word.lower())
    
# Bag of Words function: Converts a tokenized sentence into a numerical representation
def bag_of_words(tokenized_sentence, all_words):
    # Stem each word in the tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Initialize an array filled with zeros, representing the bag of words
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Populate the bag of words with 1.0 if the word is present in the tokenized sentence
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag

# Read Excel file
df = pd.read_excel('dataset.xlsx')

# Specify the path to save the CSV file
csv_file_path = 'dataset.csv'

# Save DataFrame to CSV format
df.to_csv(csv_file_path, index=False)

df = pd.read_csv('dataset.csv')
df.columns = ['Symptoms','Disease']

# Print first few rows of the DataFrame
# print(df.head())

# Count the number of unique symptoms
n_unique = len(df['Symptoms'].unique())
print(f'Number of unique symptoms: {n_unique}')

# Print data types of each column
print(df.dtypes)

# Encode the Labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['Symptoms'])
print(integer_encoded)


# One Hot Encode the Labels
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# print(onehot_encoded[0])
print(len(onehot_encoded[0]))

cols = np.asarray(df['Symptoms'].unique())
#print(cols)

# Create a new dataframe to save OHE labels
df_ohe = pd.DataFrame(columns = cols)
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]
    
# print(len(df_ohe))

# Disease Dataframe
df_disease = df['Disease']
# print(df_disease.head())

# Concatenate OHE Labels with the Disease Column
df_concat = pd.concat([df_disease,df_ohe], axis=1)
# print(df_concat.head())

df_concat.drop_duplicates(keep='first',inplace=True)

# print(len(df_concat))

cols = df_concat.columns

cols = cols[1:]

# Since, every disease has multiple symptoms, combine all symptoms per disease per row
df_concat = df_concat.groupby('Disease').sum()
df_concat = df_concat.reset_index()
df_concat[:5]

df_concat.to_csv("training_dataset.csv", index=False)

# One Hot Encoded Features
X = df_concat[cols]
print(X)
# Labels
y = df_concat['Disease']
print(y)