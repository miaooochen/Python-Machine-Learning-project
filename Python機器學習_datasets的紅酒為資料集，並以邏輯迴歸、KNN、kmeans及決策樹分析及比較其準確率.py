#以Python datasets的紅酒為資料集，並以邏輯迴歸、KNN、kmeans及決策樹分析及比較其準確率
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cluster
import warnings
warnings.filterwarnings('ignore')


wine = datasets.load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)
target = pd.DataFrame(wine.target, columns=['target'])
y1 = target['target']
y = wine.target
k= 3

kmeans = cluster.KMeans(n_clusters= k, random_state = 10)
kmeans.fit(X)
print(kmeans.labels_)
print(y)

#修正標籤 [0,2,1]
pred_y = np.choose(kmeans.labels_,[0,2,1]).astype(np.int64)
print(pred_y)
import sklearn.metrics as sm
#計算準確度 sm.accuracy_score()
print('準確度:',sm.accuracy_score(y,pred_y))


from sklearn import linear_model, neighbors, tree
logistic = linear_model.LogisticRegression()
logistic.fit(X,y1)
#print('迴歸係數:',logistic.coef_)
print('邏輯迴歸分析準確率:',logistic.score(X,y1))#是用在二元分類 故此題不適用

knn = neighbors.KNeighborsClassifier(n_neighbors= 2)
knn.fit(X,y1)
print('KNN準確率:',knn.score(X,y1))

dtree = tree.DecisionTreeClassifier(max_depth=(3))
dtree.fit(X,y1)
print('tree準確率:',dtree.score(X,y1))#準確率最高
'''
準確度: 0.702247191011236
邏輯迴歸分析準確率: 0.9662921348314607
KNN準確率: 0.8764044943820225
tree準確率: 0.9775280898876404
'''

#找最相關欄位 magnesium,hue??
import matplotlib.pyplot as plt
colmap = np.array(['r','g','y'])
plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.subplots_adjust(hspace=2)
plt.scatter(X['proanthocyanins'],X['proline'],color = colmap[kmeans.labels_])
plt.xlabel('proanthocyanins')
plt.ylabel('proline')
plt.title('kmeans classification')
plt.subplot(1,3,2)
plt.subplots_adjust(hspace=2)
plt.scatter(X['proanthocyanins'],X['proline'],color = colmap[y])
plt.xlabel('proanthocyanins')
plt.ylabel('proline')
plt.title('real classification')
plt.subplot(1,3,3)
plt.subplots_adjust(hspace=2)
plt.scatter(X['proanthocyanins'],X['proline'],color = colmap[pred_y])
plt.xlabel('proanthocyanins')
plt.ylabel('proline')
plt.title('fix kmeans classification')
plt.show()
