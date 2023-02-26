import pandas as pd

# 載入寶可夢資料集
pokemon = pd.read_csv('pokemon.csv')

# 處理遺漏值
features = ['Attack', 'Defense']
pokemon = pokemon.dropna(axis = 0, subset=features)# inplace=False

# 取出目標寶可夢的 Type1 與兩個特徵欄位
filter1 = pokemon['Type1']== 'Normal'
filter2 = pokemon['Type1']== 'Fighting'
filter3 = pokemon['Type1']== 'Ghost'
pokemon = pokemon[filter1 | filter2|filter3]
X = pokemon[features]
y = pokemon['Type1']

# 編碼 Type1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)#y轉為數字

# 特徵標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)


# 建立線性支援向量分類器，除以下參數設定外，其餘為預設值
# C=0.1, dual=False, class_weight='balanced'
from sklearn.svm import LinearSVC
model = LinearSVC(C=0.1, dual=False, class_weight='balanced')
model.fit(X_std, y)

# 計算分類錯誤的數量
predy = model.predict(X_std)
print('Misclassified samples: %d'%(predy != y).sum()) #35

# 計算準確度(accuracy)
from sklearn.metrics import accuracy_score
print('Accuracy:%.4f'% accuracy_score(y, predy)) #0.7586

# 計算有加權的 F1-score (weighted)
from sklearn.metrics import f1_score
f1 = f1_score(y,predy,average='weighted')
print('F1-score:%.4f' % f1)#0.7578

# 預測未知寶可夢的 Type1
newdata = [[100,75]]
newdata = scaler.transform(newdata)#標準化
newy = model.predict(newdata)
#print(newy) #數字
label = le.inverse_transform(newy)#反轉回字串
print(label)#Fighting 選項b