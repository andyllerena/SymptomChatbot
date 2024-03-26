import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'SymptomChatbot\Data\dataset.xlsx'


df = pd.read_excel(path)
print (df.head())

all_symptoms = []
def uniqueSymptoms(symp):
    for s in symp.split(','):
        s = s.strip()
        s = s.lower()
        if s != '' and s not in all_symptoms:
            all_symptoms.append(s)
    return symp

df['Symptoms'].apply(uniqueSymptoms)
print(len(all_symptoms))

def parse_symptoms(symptoms):
    sym = []
    for s in all_symptoms:
        if s in symptoms:
            sym.append(1)
        else:
            sym.append(0)
    return sym

data = []
sym = []
for row in df.iterrows():
    sym = parse_symptoms(row[1].drop(columns='disease'))
    sym.append(row[1]['Disease'])
    data.append(sym)

final_df = pd.DataFrame(data,columns=all_symptoms+["Disease"])

print(final_df.shape)