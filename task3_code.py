# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:48:40 2020

@author: Jessica
"""

# Word Embedding (词嵌入), 将不定长的文本转换到定长的空间内

# One-hot 每一个单词使用一个离散的向量表示, 具体将每个字/词编码一个索引，然后根据索引进行赋值
# ex. sen 1: 我喜欢李钟硕  sen 2: 我爱大海
# step 1:确定字编号. 
#{
#	'我': 1, '喜': 2, '欢': 3, '李': 4, '钟': 5,
#  '硕': 6, '爱': 7, '大': 8, '海'：9
#}
# Transform word to vector: 我：[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; 喜：[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0].......

# Bag of Words（词袋表示），也称为Count Vectors，每个文档的字/词可以使用其出现次数来进行表示
# ex: sen 1:[1,1,1,1,1,1,0,0,0]; sen 2:[1,0,0,0,0,0,1,1,1]


# N-gram与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数; 如果N=2,就是相邻字分别和其相邻字连在一起
# ex:sen 1: 我喜 喜欢 欢李 李钟 钟硕; sen 2: 我爱 爱大 大海

# TF-IDF 分数由两部分组成：第一部分是词语频率（Term Frequency），第二部分是逆文档频率（Inverse Document Frequency)
# 其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率: 
# TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
# IDF(t)= log_e（文档总数 / 出现该词语的文档总数）


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
clf.fit(train_test[:10000], train['label'].values[:10000])    # fit(X,Y)

# Prediction
val_pred = clf.predict(train_test[10000:])
print(f1_score(train['label'].values[10000:], val_pred, average='macro'))
# score=0.65

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)  # ngram_range=(1,3)-> min:unigram, max: trigram
train_test = tfidf.fit_transform(train['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train['label'].values[:10000])

val_pred = clf.predict(train_test[10000:])
print(f1_score(train['label'].values[10000:], val_pred, average='macro'))


