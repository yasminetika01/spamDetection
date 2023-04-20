import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

df = pd.read_csv("data/cleaned_transformed_data.csv")

X = tfidf.fit_transform(df['transformed_text'].values.astype('U')).toarray()
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X.shape
y = df['target'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

lr = LogisticRegression(solver='liblinear',penalty='l1')
knn = KNeighborsClassifier()
mnb = MultinomialNB()

clfs = {
    'LR' : lr,
    'KN' : knn,
    'NB' : mnb
    }

accuracy_scores = []
precision_scores = []
for nom,cle in clfs.items():
    current_accuracy,current_precision = train_classifier(cle,X_train,y_train,X_test,y_test)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_fig = plt.figure(figsize=(15,15))
performance_df = pd.DataFrame(list(zip(accuracy_scores,precision_scores, clfs.keys())), columns=["Accuracy", "Precision", "Algorithm"])
sns.catplot(col="Algorithm", data = performance_df, kind = 'bar', height = 5)

pickle.dump(mnb, open('models/model1.pkl', 'wb'))

def app():
    st.subheader("Model Building")
    st.dataframe(df.head())

    performance_df = pd.DataFrame(list(zip(accuracy_scores,precision_scores, clfs.keys())), columns=["Accuracy", "Precision", "Algorithm"])
    sns.catplot(col="Algorithm", data = performance_df, kind = 'bar', height = 5)
    st.pyplot()
