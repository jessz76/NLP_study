# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:48:40 2020

@author: Jiaxin
"""

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()
vectorizer.fit_transform(corpus).toarray()

import pandas as pd
train = pd.read_csv(r'E:\MMA summer2020\NLP_datawhale\train.csv\train_set.csv',sep='\t', nrows=15000)

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train['label'].values[10000:], val_pred, average='macro'))
# score=0.65

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train['label'].values[10000:], val_pred, average='macro'))


