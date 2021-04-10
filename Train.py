import numpy as np
import pandas as pd

import bz2

trainfile = bz2.BZ2File('./Dataset/train.ft.txt.bz2','r')
lines = trainfile.readlines()


docSentimentList=[]
def getDocumentSentimentList(docs,splitStr='__label__'):
    for i in range(len(docs)):
        text=str(lines[i])
        splitText=text.split(splitStr)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        docSentimentList.append([text,sentiment])
    print('OK')
    return docSentimentList

docSentimentList=getDocumentSentimentList(lines[:1000000],splitStr='__label__')

train_df = pd.DataFrame(docSentimentList,columns=['Review','Sentiment'])

"""## **Text Preprocessing**##"""

train_df['Sentiment'][train_df['Sentiment']=='1'] = 0
train_df['Sentiment'][train_df['Sentiment']=='2'] = 1

train_df['word_count'] = train_df['Review'].str.lower().str.split().apply(len)

import string 
def remove_punc(s):
    table = str.maketrans({key: None for key in string.punctuation})
    return s.translate(table)

train_df['Review'] = train_df['Review'].apply(remove_punc)

train_df1 = train_df[:][train_df['word_count']<=25]

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
st_wd = text.ENGLISH_STOP_WORDS
c_vector = CountVectorizer(stop_words = st_wd,min_df=.0001,lowercase=1)
c_vector.fit(train_df1['Review'].values)

word_list = list(c_vector.vocabulary_.keys())
stop_words = list(c_vector.stop_words)

def remove_words(raw_sen,stop_words):
    sen = [w for w in raw_sen if w not in stop_words]
    return sen

def reviewEdit(raw_sen_list,stop_words):
    sen_list = []
    for i in range(len(raw_sen_list)):
        raw_sen = raw_sen_list[i].split()
        sen_list.append(remove_words(raw_sen,stop_words))
    return sen_list

sen_list = reviewEdit(list(train_df1['Review']),stop_words)

from gensim.models import word2vec
wv_model = word2vec.Word2Vec(sen_list,size=100)

def fun(sen_list,wv_model):
    word_set = set(wv_model.wv.index2word)
    X = np.zeros([len(sen_list),25,100])
    c = 0
    for sen in sen_list:
        nw=24
        for w in list(reversed(sen)):
            if w in word_set:
                X[c,nw] = wv_model[w]
                nw=nw-1
        c=c+1
    return X

X = fun(sen_list,wv_model)

from sklearn.model_selection import train_test_split
y = train_df1['Sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train = y_train.astype('bool')
y_test = y_test.astype('bool')

""" ## **Keras NN Model** ##"""

from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,LSTM, SimpleRNN ,GRU , Bidirectional,Input ,Concatenate, Multiply,Lambda,Reshape
input_st  = Input(shape=(25,100))
lstm1 = Bidirectional(LSTM(200,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(input_st)
lstm2 = Bidirectional(LSTM(1,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(lstm1)
lstm2 = Reshape((-1,))(lstm2)
lstm2 = Activation('sigmoid')(lstm2)
lstm2 = Reshape((-1,1))(lstm2)
mult = Multiply()([lstm1,lstm2])

add = Lambda(lambda x: K.sum(x,axis=1))(mult)
dense = Dense(100,activation='relu')(add)
output = Dense(1,activation='sigmoid')(dense)

model = Model(inputs=input_st, outputs=output)
print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train,y_train,validation_split=0.1,
          epochs=10, batch_size=512)

y_test = y_test.astype('bool')
model.evaluate(X_test, y_test, batch_size=64)

model.save("./Models/BidirectionalLSTM")

del model

input_st  = Input(shape=(25,100))
lstm1 = Bidirectional(GRU(200,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(input_st)
lstm2 = Bidirectional(GRU(1,input_shape=(25,100),activation='relu',return_sequences=True),merge_mode='mul')(lstm1)
lstm2 = Reshape((-1,))(lstm2)
lstm2 = Activation('sigmoid')(lstm2)
lstm2 = Reshape((-1,1))(lstm2)
mult = Multiply()([lstm1,lstm2])

add = Lambda(lambda x: K.sum(x,axis=1))(mult)
dense = Dense(100,activation='relu')(add)
output = Dense(1,activation='sigmoid')(dense)

model = Model(inputs=input_st, outputs=output)
print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X_train,y_train,validation_split=0.1,
          epochs=10, batch_size=512)

model.evaluate(X_test, y_test, batch_size=64)

model.save("./Models/BidirectionalGRU")