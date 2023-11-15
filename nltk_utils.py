import nltk 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
    
def bag_of_words(tokenized_sentence,all_word):
    pass

a = "Hello how are you"
print(a)
a = tokenize(a)
print(a)

words = ["connect","connected","connecting", "connection","connections"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)

