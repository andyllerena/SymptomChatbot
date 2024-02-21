# Import necessary libraries
import nltk 
from nltk.stem.porter import PorterStemmer
import numpy as np

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








