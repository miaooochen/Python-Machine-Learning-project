#以Python datasets的iris為資料集，並以KNN及決策樹(訓練集測試集)分析及比較其準確率
from sklearn import datasets
from sklearn import neighbors, tree
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns= iris.feature_names)
y = pd.DataFrame(iris.target, columns= ['target'])
k = 50
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(X,y)
print('全部資料 KNN準確率:',knn.score(X,y))

dtree = tree.DecisionTreeClassifier(max_depth=(4))
dtree.fit(X,y)
print('全部資料 決策樹準確率:',dtree.score(X,y))


from sklearn.model_selection import train_test_split as tts

XTrain,XTest,yTrain,yTest = tts(X,y,test_size=0.4,random_state=(50))
rate1 =0
k1 =0
for i in range(1,91):
    
    knn2 = neighbors.KNeighborsClassifier(n_neighbors= i)
    knn2.fit(XTrain,yTrain)
    rate2= knn2.score(XTest,yTest)
    if rate2 > rate1:
        rate1 = rate2
        k1 =i
print('測試集 KNN準確率:',rate1, 'K=',k1)
'''全部資料 KNN準確率: 0.94
全部資料 決策樹準確率: 0.9933333333333333
測試集 KNN準確率: 0.9833333333333333 K= 3
'''