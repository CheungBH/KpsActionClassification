from sklearn.model_selection import cross_val_score
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# ====================Gradient Tree Boosting（梯度树提升）=========================
# 分类
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
scores = cross_val_score(clf, X, y)
print('GDBT准确率：', scores.mean())

# 回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

boston = load_boston()  # 加载波士顿房价回归数据集
X1, y1 = shuffle(boston.data, boston.target, random_state=13)  # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1,
                                                    random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例
clf = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, min_samples_split=2, loss='ls')
clf.fit(X1, y1)
print('GDBT回归MSE：', mean_squared_error(y_test, clf.predict(X_test)))
# print('每次训练的得分记录：',clf.train_score_)
print('各特征的重要程度：', clf.feature_importances_)
plt.plot(np.arange(500), clf.train_score_, 'b-')  # 绘制随着训练次数增加，训练得分的变化
plt.show()
