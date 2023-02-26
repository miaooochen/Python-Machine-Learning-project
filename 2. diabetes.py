from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts 

dia = datasets.load_diabetes()
X = pd.DataFrame(dia.data,  columns= dia.feature_names)
y = pd.DataFrame(dia.target, columns=['diabetes'])

lm = LinearRegression()
lm.fit(X,y)
predy = lm.predict(X)
MSE = mse(y,predy)

print('Total number of examples')
print('MSE=',MSE.round(4))
print('R-squared=',lm.score(X,y).round(4))
#3:1 100
XTrain, XTest, yTrain, yTest=tts(X, y, test_size=0.25, random_state=(100))
lm2=LinearRegression()
lm2.fit(XTrain,yTrain)
predytest = lm2.predict(XTest)
predytrain = lm2.predict(XTrain)

print('Split 3:1')
print('train MSE=',mse(yTrain,predytrain).round(4))
print('test MSE=', mse(yTest,predytest).round(4))
print('train R-squared=',lm.score(XTrain,yTrain).round(4))
print('test R-squared=',lm.score(XTrain,yTrain).round(4))
'''
Total number of examples
MSE= 2859.6904 
R-squared= 0.5177
Split 3:1
train MSE= 2947.9337
test MSE= 2665.2278
train R-squared= 0.5035
test R-squared= 0.5035
'''