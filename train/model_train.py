import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle
import os

train_path = "C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\Data\\training.csv"
test_path = "C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\Data\\testing.csv"

df = pd.read_csv(train_path)
print(df.head())

tdf = pd.read_csv(test_path)
print(tdf.head())


X = df.drop(columns=['disease'],axis=0).values
y = df['disease'].values
print (len(X), len)

X_test = tdf.drop(columns=['disease'],axis=0).values
y_test = tdf['disease'].values
print(len(X_test),len(y_test))

def evaluate_model(model):
    y_pred = model.predict(X_test)
    print('Accuracy: ',accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

rf = RandomForestClassifier()

rf.fit(X,y)

dtc = DecisionTreeClassifier()
dtc.fit(X,y)

print(evaluate_model(dtc))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

print(evaluate_model(knn))

svc = SVC()
svc.fit(X,y)

print(evaluate_model(svc))

def save_model(model,name,version):
    dir_path = "C:\\Users\\aller\\Desktop\\chatbot3\\SymptomChatbot\\models"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_path = os.path.join(dir_path,'v'+version)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    path = os.path.join(dir_path,f"{name}-v{version}.pkl")
    with open(path,'wb') as f:
        pickle.dump(model,f)

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

v = '2'
save_model(rf,f'random_forest',v)
save_model(svc,f'svc',v)
save_model(dtc,f'decision_tree',v)
save_model(knn,f'knn',v)