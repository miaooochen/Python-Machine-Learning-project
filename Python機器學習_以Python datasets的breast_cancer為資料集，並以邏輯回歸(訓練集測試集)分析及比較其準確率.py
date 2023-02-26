#以Python datasets的breast_cancer為資料集，並以邏輯回歸(訓練集測試集)分析及比較其準確率
import pandas as pd
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts

breast_cancer = datasets.load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
y = pd.DataFrame(breast_cancer.target, columns = ['target'])
#y = breast_cancer.target (type=ndarray)
logistic = linear_model.LogisticRegression()
logistic.fit(X,y) #一維二維都可建立模型
print('全部資料正確率:',logistic.score(X, y))

XTrain, XTest, yTrain, yTest = tts(X,y,test_size=0.3,random_state=10)
logistic = linear_model.LogisticRegression()
logistic.fit(XTrain,yTrain)
#print('迴歸係數:',logistic.coef_)
print('訓練集正確率:',logistic.score(XTrain,yTrain))
print('測試集正確率:',logistic.score(XTest,yTest))
#也可用 sort_values

newX = pd.DataFrame([X.iloc[:,0],X.iloc[:,20],X.iloc[:,21],X.iloc[:,25],X.iloc[:,26]]).T
logistic = linear_model.LogisticRegression()
logistic.fit(newX,y)
print('迴歸係數前五高之正確率:',logistic.score(newX, y))
print('''迴歸係數越接近零,相關性越低,
所以選擇了迴歸係數前五大的五個欄位建立模型,
正確率相比之下也較高''')