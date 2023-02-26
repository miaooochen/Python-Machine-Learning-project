#以Python datasets的diabetes為資料集，並以線性回歸(訓練集測試集)分析及比較其準確率
from sklearn import datasets
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
target = pd.DataFrame(diabetes.target, columns = ['diabetes'])
y = target['diabetes']

lm=LR()
lm.fit(X,y)
pred_y = lm.predict(X)
MSE = np.mean((y-pred_y)**2)
print('=====不分割資料集=====')
print('MSE:',MSE)
print('R-squared:',lm.score(X,y))
print()

XTrain, XTest, yTrain, yTest = tts(X,y,test_size=0.25, random_state=100)
lm= LR()
lm.fit(XTrain, yTrain)
pre_yTrain = lm.predict(XTrain)
pre_yTest = lm.predict(XTest)
print('==資料集分割比率 3:1 ==')
print('訓練資料集 MSE:',mse(yTrain, pre_yTrain))
print('測試資料集 MSE:',mse(yTest, pre_yTest))
print('訓練資料集 R-squared:',lm.score(XTrain,yTrain))
print('測試資料集 R-squared:',lm.score(XTest,yTest))
print()

XTrain, XTest, yTrain, yTest = tts(X,y,test_size=0.20, random_state=100)
lm= LR()
lm.fit(XTrain, yTrain)
pre_yTrain = lm.predict(XTrain)
pre_yTest = lm.predict(XTest)
print('==資料集分割比率 4:1 ==')
print('訓練資料集 MSE:',mse(yTrain, pre_yTrain))
print('測試資料集 MSE:',mse(yTest, pre_yTest))
print('訓練資料集 R-squared:',lm.score(XTrain,yTrain))
print('測試資料集 R-squared:',lm.score(XTest,yTest))
print()

print('''選擇第 3 種(資料集分割比率 4:1)方式建立模型,因為 MSE誤差值最小''')
