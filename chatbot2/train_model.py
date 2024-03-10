from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import data_preparation

# Load the dataset
file_path = "C:\\Users\\aller\\Desktop\\chatbot\\chatbot2\\training_dataset.csv"
df = pd.read_csv(file_path)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(data_preparation.X, data_preparation.y, test_size=0.2, random_state=101)
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))

dt = DecisionTreeClassifier()
clf_dt=dt.fit(data_preparation.X, data_preparation.y)

print(clf_dt.score(data_preparation.X, data_preparation.y))

# Evaluation on test data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test set
y_pred = clf_dt.predict(X_test)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))