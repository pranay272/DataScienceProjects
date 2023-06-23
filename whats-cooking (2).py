import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

with open("C:/Users/parab/OneDrive/Desktop/p3/whats-cooking (1)/train.json/train.json") as file:
    data = json.load(file)

df = pd.DataFrame(data)

print(df.head(10))

print(df.columns)

print(df.describe())

print(df['cuisine'].nunique())

#print(df.isnull().sum())

df.dropna(inplace=True)

# df['ingredients'] = df['ingredients'].apply(','.join)
# df.drop_duplicates(inplace=True)
#print({df.duplicated().sum()})

empty_documents = df[df['ingredients'].apply(lambda x: len(x) == 0)]
if not empty_documents.empty:
    print("Empty documents found!")
    # Handle empty documents here, such as dropping or filling them
else:
    print("No empty documents found.")

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['ingredients'].apply(' '.join))

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(features, df['cuisine'], test_size=0.2, random_state=42)

# (Multinomial Naive Bayes)
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("----------------------------------------------------------")
print("Multinomial Naive Bayes Accuracy:", accuracy)
print("----------------------------------------------------------")
print(classification_report(y_test, y_pred))


# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("------------------------------------------------------")
print("Logistic Regression Accuracy:", accuracy_lr)
print("------------------------------------------------------")
print(classification_report(y_test, y_pred_lr))


# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("-----------------------------------------------------")
print("Random Forest Accuracy:", accuracy_rf)
print("-----------------------------------------------------")
print(classification_report(y_test, y_pred_rf))

# Support Vector Machines (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("-----------------------------------------------------")
print("SVM Accuracy:", accuracy_svm)
print("-----------------------------------------------------")
print(classification_report(y_test, y_pred_svm))
