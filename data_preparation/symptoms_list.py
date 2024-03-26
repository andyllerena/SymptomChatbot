import pandas as pd
import numpy as np

path = 'C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\Data\\raw_data.xlsx'

df = pd.read_excel(path)

print(df.head())

diseases = []
disCount = []
symps = []
prevDis = None
sym = []
for row in df.iterrows():
    dis = row[1]['Disease']
    if not pd.isna(dis):
        # print('hi')
        if row[0] != 0:
            symps.append(sym)
        sym = []
        diseases.append(dis)
        disCount.append(row[1]['Count of Disease Occurrence'])
        prevDis = dis
    sym.append(row[1]['Symptom'])
symps.append(sym)

print(len(diseases),len(disCount),len(symps))

data = pd.DataFrame({"disease":diseases,"count":disCount,"symptoms":symps})

type(data['symptoms'][0][0])
"|".join(data['symptoms'][0])

def parseSymptoms(syms):
    print(type(syms))
    # for sy in syms:
    #     print(sy)

print(data['symptoms'].apply(lambda x: print(x)))

print(data['symptoms'][0])