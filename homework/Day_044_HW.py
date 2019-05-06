
# coding: utf-8

# ## 作業
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較



from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split




# 讀取紅酒資料集
wine = datasets.load_wine()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=4)

# 建立模型
clf = RandomForestClassifier(
    n_estimators=10, #決策樹的數量
    criterion="gini",
    max_features="auto", #如何選取 features
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)




acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)




print(wine.feature_names)




print("Feature importance: ", clf.feature_importances_)



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# 回歸模型
# 讀取紅酒資料集
wine = datasets.load_wine()

# # 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, random_state=4)

# # 建立一個線性回歸模型
regr = linear_model.LogisticRegression()

# # 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)

# # 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)




# 可以看回歸模型的參數值
print('Coefficients: ', regr.coef_)

# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))




from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# 決策樹
# 讀取紅酒資料集
wine = datasets.load_wine()

# # 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.1, random_state=4)

# 建立模型
clf = DecisionTreeClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)




acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)

