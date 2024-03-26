import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'SymptomChatbot\Data\dataset.xlsx'

df = pd.read_excel(path)
print (df['Disease'].unique())

print(df.Disease.nunique())

symptoms = []
def uniqueSymptoms(symp):
    for s in symp.split(','):
        s = s.strip()
        s = s.lower()
        if s != '' and s not in symptoms:
            symptoms.append(s)
    return symp

print(df['Symptoms'].apply(uniqueSymptoms))

print(len(symptoms))

print(symptoms)

print(df.duplicated().sum())

pd.set_option('display.max_colwidth', 0)

print(df[df.duplicated()])

