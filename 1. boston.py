from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
#from sklearn.metrics import root_mean_squared_error as rmse
import pandas as pd

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns = ['MEDV'])

XTrain, XTest, yTrain,yTest = tts(X,y,train_size=0.2,random_state=(1))
lm = linear_model.LinearRegression()
lm.fit(XTrain,yTrain)
pred_yTest = lm.predict(XTest)
MSE = mse(yTest,pred_yTest)
MAE = mae(yTest,pred_yTest)
RMSE = mse(yTest,pred_yTest)**0.5

print('MAE:', MAE.round(4))
print('MSE:', MSE.round(4))
print('RMSE:', RMSE.round(4))

X_new =[[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 
         65.20, 4.0900, 1, 296.0, 15.30, 396.90 , 4.98]]
newX = pd.DataFrame(X_new)#建模資料類型及維度需要一致
pred_y = lm.predict(newX)
print(pred_y.round(4))
'''
MAE: 3.4962
MSE: 23.1837
RMSE: 4.8149
[[30.5479]]
'''