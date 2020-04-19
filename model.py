# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:09:00 2020

@author: Siva Rami Reddy
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

data = pd.read_csv("H:\ML pratice\Mobile catigory predection\mobile.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]  
data.head()

data.describe(include='all')

# Finding the unique values of the data of catigorical values
for col_names in data.columns:
    if data[col_names].dtype=='object':
        col_uniq= data[col_names].unique().sum()
        print(f"{col_names} has {col_uniq} values")
        
#apply SelectKBest class to extract top 10 best features to select the best featues for getting the better accuracy
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores

print(featureScores.nlargest(10,'Score'))  #print 10 best features

X=data.loc[:,['ram','px_height','battery_power','px_width','mobile_wt','int_memory','sc_w','talk_time','fc','sc_h']]
X.head()

X.isnull().sum()

to_dummy_list=[]
def dummy_df(df,to_dummy_list):
    for x in to_dummy_list:
        dummies=pd.get_dummies(df[x],prefix=x,dummy_na=False)
        df=df.drop(x,1)
        df=pd.concat([df,dummies],axis=1)
    return df

X=dummy_df(X,to_dummy_list)
X.head()

# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=44)

# Finding which alogrithem gives the best fit and gives the better accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data.hist(figsize=(14,13))
plt.show()

seed=6
models=[]
scoring='accuracy'

models.append(('LR',LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('Lart',DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(('SVM',SVC()))
from sklearn import model_selection

results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"This {name} model - has {int(cv_results.mean()*100)}% accuracy - {cv_results.std()}")

# Based on the best accuracy select the algorithem for the building the model and build the model
# in this LinaerDiscriminateAnalysis gives the better accuracy when compared with the other algorithems
linearDA=LinearDiscriminantAnalysis()

linearDA.fit(X_train,y_train)

y_pred=linearDA.predict(X_test)
X_test.info()

confusion_matrix(y_test,y_pred)

accuracy=int(accuracy_score(y_test,y_pred)*100)
print(f"The total accuracy is {accuracy}")

import seaborn as sns
corrmat=data.corr()
top_corr_fetures=corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_fetures].corr(),annot=True,cmap="RdYlGn")
plt.show()

import pickle
pickle.dump(linearDA, open('model.pkl','wb'))

# Loading model to compare the results
predection = pickle.load(open('model.pkl','rb'))
print(predection.predict([[2631, 905,1021,1988,136,53,3,7,0,8]]))
print(X.columns)