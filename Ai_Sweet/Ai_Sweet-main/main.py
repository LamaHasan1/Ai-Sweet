
import pandas as pd
import re
import csv

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
sklearn.metrics.classification_report


import numpy as np
import exFeatures

# *******
print("************* Automation Sweet spam *************")


def compareFwingFowes(fwing, fowers):
    ff = []
    for index in range(len(fwing)):
        if fwing[index] > fowers[index]:
            ff.append(1)
        else:
            ff.append(0)

    print(ff)
    return ff


def CharactersofConten():
    CharaToSpam = []
    regularExpersion = r'//t.co/'
    for i in range(len(dataset['Tweet'])):
        temp = re.findall(regularExpersion, dataset['Tweet'].values[i])
        if len(temp) > 0 :
            CharaToSpam.append(1)
        else:
            CharaToSpam.append(0)

    print(CharaToSpam)
    return CharaToSpam

def counthashtag():
    hashtages = []
    regularExpersion = r'#'
    for i in range(len(dataset['Tweet'])):
        temp = re.findall(regularExpersion, dataset['Tweet'].values[i])
        hashtages.append(len(temp))


    print(hashtages)
    return hashtages

def countmentiones():
    mentions = []
    regularExpersion = r'@'
    for i in range(len(dataset['Tweet'])):
        temp = re.findall(regularExpersion, dataset['Tweet'].values[i])
        mentions.append(len(temp))


    print(mentions)
    return mentions


def CharactersofLocation():
    location = []
    reg = r'United States|[:>#@!,?]+'
    for ii in range(len(dataset['location'])):
        s = re.findall(reg, dataset['location'].values[ii])
        location.append(len(s))
        print(s)
    return location

def countURL():
    urls =[]
    regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    regex2 = r'[a-zA-Z]\.[a-zA-Z]+.\.com\/(?:[\da-zA-Z]{2})+'
    for i in range(len(dataset['Tweet'])):
        url1 = re.findall(regex, dataset['Tweet'].values[i])
        url2 = re.findall(regex2, dataset['Tweet'].values[i])
        # print(url1)
        # print(url2)
        print(len(url1) + len(url2))
        urls.append(len(url1) + len(url2))
        # print("**********")
    return urls

# read data set
dataset = pd.read_csv('train.csv')
print(dataset)
#print(dataset.columns)


features = []

#print(dataset.values)
#for row in range(len(dataset)):


#replace null values



dataset['following'].fillna(value=np.round(np.mean(dataset['following'])),inplace=True)
dataset['followers'].fillna(value=np.round(np.mean(dataset['followers'])),inplace=True)
dataset['actions'].fillna(value=np.round(np.mean(dataset['actions'])),inplace=True)
dataset['is_retweet'].fillna(value=np.round(np.mean(dataset['is_retweet'])),inplace=True)
dataset['location'].fillna(value='un Known',inplace=True)

following = dataset['following']
followers = dataset['followers']


following = compareFwingFowes(following,followers)

contentchara = CharactersofConten()
hashtages = counthashtag()
mention = countmentiones()
locations = CharactersofLocation()
numberofurl = countURL()


# section to write extract features to csv file
import os.path

if os.path.exists('Vector.csv') == False :
    vectorofvetures = [[]]
    vectorofvetures[0] =[contentchara[0], numberofurl[0], hashtages[0], mention[0], following[0],
             locations[0], dataset['actions'][0], dataset['is_retweet'][0], dataset['Type'][0]]
    # vectorofvetures.append(['conten_bad','urls','hashtags','mentions','following','locations_bad','actions','is_retweet','Type'])
    for index in range(1,len(dataset['Id'])):
        vectorofvetures.append(
            [contentchara[index], numberofurl[index], hashtages[index], mention[index], following[index],
             locations[index], dataset['actions'][index], dataset['is_retweet'][index], dataset['Type'][index]])
    extract_features = pd.DataFrame(vectorofvetures,
                                    columns=['conten_bad', 'urls', 'hashtags', 'mentions', 'following', 'locations_bad',
                                             'actions', 'is_retweet', 'Type'])
    print(vectorofvetures[1])
    extract_features.to_csv('Vector.csv', index=False)


dataset2 = pd.read_csv('Vector.csv')
xx = dataset2.columns[dataset2.isna().any()]
print(dataset2)
print(xx)

x = dataset2.drop(columns=['Type'])
y = dataset2['Type']
print(x)


#training and testing

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)


predicitions = model.predict(x_test)
score = accuracy_score(y_test,predicitions)
print("the Accuracy for my model use decision tree is :")
print(score)
print("DecisionTree report" )
print(classification_report(y_test,predicitions))
#print(predicitions)

#****************
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model =naiveBayes=GaussianNB()
naiveBayesModelResul=naiveBayes.fit(x_train,y_train)

predicitions = naiveBayesModelResul.predict(x_test)
score = accuracy_score(y_test,predicitions)
print("the Accuracy for my model use Naive Bayes is :")
print(score)
print(" naiveBayes report" )
print(classification_report(y_test,predicitions))
#**************************
'''
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(x_train,y_train)
predicitions = model.predict(x_test)
score = accuracy_score(y_test,predicitions)
print("the Accuracy for my model use neuralnetwork is :")
print(score)
'''
#**********************
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
mlp = MLPClassifier(hidden_layer_sizes=(), activation='tanh', solver='sgd', max_iter=600)
model=mlp.fit(X_train,y_train)
predicitions = model.predict(X_test)
score = accuracy_score(y_test,predicitions)
print("the Accuracy for my model use neuralnetwork is :")
print(score)
print("neuralnetwork report" )
print(classification_report(y_test,predicitions))
