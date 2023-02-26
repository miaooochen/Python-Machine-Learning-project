import numpy as np
from sklearn import metrics
#from sklearn.metrics import silhouette_score as ss
from sklearn import cluster


#載入資料
X = []
with open('data_perf.txt','r')as file:
    for line in file.readlines():
        data = [float(i) for i in line.split(',')]
        X.append(data)
X = np.array(X)
#載入資料
newx=[]
with open('data_perf_add.txt','r')as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        newx.append(data)
newx = np.array(newx)

# Find the best epsilon半徑 
eps_grid = np.linspace(0.3,1.2,num=10)
silhouette_scores = []
eps_best= eps_grid[0]
silhouette_scores_max = -1 #越接近1越好
labels_best = None
model_best = None

for eps in eps_grid:
    # Train DBSCAN clustering model 訓練DBSCAN分群模型 # min_samples = 5
    dbscan = cluster.DBSCAN(eps= eps, min_samples=5)
    model = dbscan.fit(newx)
    # Extract labels 提取標籤
    labels = model.labels_
    # Extract performance metric 提取性能指標
    silhouette_score = metrics.silhouette_score(newx,labels).round(4)
    silhouette_scores.append(silhouette_score)

    print("Epsilon:", eps, " --> silhouette score:", silhouette_score)
    if silhouette_score > silhouette_scores_max:
        silhouette_scores_max = silhouette_score
        eps_best = eps
        labels_best = labels
        model_best = model
'''
Epsilon: 0.3  --> silhouette score: 0.1287
Epsilon: 0.39999999999999997  --> silhouette score: 0.3594
Epsilon: 0.5  --> silhouette score: 0.5134
Epsilon: 0.6  --> silhouette score: 0.6165
Epsilon: 0.7  --> silhouette score: 0.6322
Epsilon: 0.7999999999999999  --> silhouette score: 0.6366
Epsilon: 0.8999999999999999  --> silhouette score: 0.5142
Epsilon: 1.0  --> silhouette score: 0.5629
Epsilon: 1.0999999999999999  --> silhouette score: 0.5629
Epsilon: 1.2  --> silhouette score: 0.5629
'''
    
print("Best epsilon =",eps_best)
print('best silhouette score:', silhouette_scores_max)
finalmodel = model_best
finallabels =labels_best

# Check for unassigned datapoints in the labels離群值
offset = 0
if -1 in finallabels:
    offset = 1

num = len(set(finallabels))-1 #轉為set扣除重複
print("Estimated number of clusters =",num)
'''
Best epsilon = 0.7999999999999999
Estimated number of clusters = 5
'''


# Extracts the core samples from the trained model

'''讀取第二份檔案後
Epsilon: 0.3  --> silhouette score: 0.152
Epsilon: 0.39999999999999997  --> silhouette score: 0.3631
Epsilon: 0.5  --> silhouette score: 0.5188
Epsilon: 0.6  --> silhouette score: 0.6205
Epsilon: 0.7  --> silhouette score: 0.6358
Epsilon: 0.7999999999999999  --> silhouette score: 0.6401
Epsilon: 0.8999999999999999  --> silhouette score: 0.5155
Epsilon: 1.0  --> silhouette score: 0.5641
Epsilon: 1.0999999999999999  --> silhouette score: 0.5641
Epsilon: 1.2  --> silhouette score: 0.5641

Best epsilon = 0.7999999999999999
best silhouette score: 0.6401
Estimated number of clusters = 5
'''

