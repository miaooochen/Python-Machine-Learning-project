# #############################################################################
# 本題參數設定，請勿更改
seed = 0    # 亂數種子數
# #############################################################################

import numpy as np
#from sklearn.datasets import load_digits
from sklearn import datasets

# 載入手寫數字資料集
digit = datasets.load_digits()
X_digits = digit.data  
y_digits = digit.target

# 特徵標準化(scale/StandardScaler)
from sklearn.preprocessing import scale
#from sklearn.preprocessing import StandardScaler #須建立標準化物件
data = scale(X_digits)

# 取出資料集的數字類別數
n_digits = 10

# 建立兩個 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# kmean1: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# kmean2: init='random', n_clusters=n_digits, n_init=10, random_state=seed
# #############################################################################
from sklearn.cluster import KMeans
kmeans1 = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed)
kmeans2 = KMeans(init='random', n_clusters=n_digits, n_init=10, random_state=seed)

# 利用 PCA 結果建立 K-Means 模型，除以下參數設定外，其餘為預設值
# #############################################################################
# pca: n_components=n_digits, random_state=seed
# kmean3: init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed
# #############################################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=n_digits, random_state=seed).fit(data)
kmeans3 = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1, random_state=seed)

# 分別計算上述三個 K-Means 模型的輪廓係數(Silhouette coefficient)與
# 分類準確率(accuracy)，除以下參數設定外，其餘為預設值
# #############################################################################
# silhouette_score: metric='euclidean'
# #############################################################################
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
lst_name = ['K-Mean (k-means++)', 'K-Means (random)', 'K-Means (PCA-based)']
lst_model = [kmeans1, kmeans2, kmeans3]
for name, model in zip(lst_name, lst_model):
    model.fit(data)
    silhouette = silhouette_score(data, model.labels_, metric='euclidean')
    print('%s: Silhouette= %.4f'% (name,silhouette),end='')
    print(' Accrucy:%.4f'%(accuracy_score(y_digits, model.labels_)))
    
# 進行 PCA 降維後再做 K-Means，除以下參數設定外，其餘為預設值
# #############################################################################
# kmeans: init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed
# PCA: n_components=2, random_state=seed
# #############################################################################
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, random_state=seed)
reduced_data = PCA(n_components=2, random_state=seed).fit_transform(data)#
kmeans.fit(reduced_data)
print('PCA+KMeans Silhouette= %.4f'% silhouette_score(reduced_data, model.labels_,metric='euclidean'))
print('Accuracy= %.4f'% accuracy_score(y_digits, kmeans.labels_))
'''
K-Mean (k-means++): Silhouette= 0.1441 Accrucy:0.1146
K-Means (random): Silhouette= 0.1410 Accrucy:0.0351
K-Means (PCA-based): Silhouette= 0.1388 Accrucy:0.1085
PCA+KMeans Silhouette= 0.0978
Accuracy= 0.2065
'''