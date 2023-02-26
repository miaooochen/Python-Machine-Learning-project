import pandas as pd
import numpy as np

titanic = pd.read_csv('titanic.csv')
age_median = np.nanmedian(titanic['Age'])
print('年齡中位數:',age_median)
new_age = np.where(titanic['Age'].isnull(),age_median,titanic['Age'])
titanic['Age']= new_age

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
encoded_pclass = le.fit_transform(titanic['PClass'])
X = pd.DataFrame([encoded_pclass, titanic['Age'], titanic['SexCode']]).T
y = titanic['Survived']

from sklearn import linear_model
logistic = linear_model.LogisticRegression()
logistic.fit(X,y)
print('SexCode迴歸係數:',logistic.coef_[0][2].round(4))
print('截距:',logistic.intercept_.round(4))
print('準確率',logistic.score(X,y).round(4))
'''
年齡中位數: 28.0
SexCode迴歸係數: 2.3834
截距: [1.9966]
準確率 0.8149
'''