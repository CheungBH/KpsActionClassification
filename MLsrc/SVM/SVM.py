from sklearn import svm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import dump


feature_num = 26
data=[]
traffic_feature=[]
traffic_target=[]

csv_file = csv.reader(open("../tennis_player_4cls.csv"))
for content in csv_file:
    content=list(map(float,content[:-1]))
    if len(content)!=0:
        data.append(content)
        traffic_feature.append(content[0:feature_num])
        traffic_target.append(content[feature_num])
# print('data=',data)
# print('traffic_feature=',traffic_feature)
# print('traffic_target=',traffic_target)

scaler = StandardScaler() # 标准化转换
scaler.fit(traffic_feature)  # 训练标准化对象
traffic_feature= scaler.transform(traffic_feature)   # 转换数据集

feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3, random_state=0)
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf = svm.SVC(C=10, kernel='poly', decision_function_shape='ovr')
clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')


clf.fit(feature_train,target_train)
predict_results=clf.predict(feature_test)
dump(clf, '../../exp/SVM/SVM_test.joblib')

print("Testing accuracy is {}\n".format(accuracy_score(predict_results, target_test)))
conf_mat = confusion_matrix(target_test, predict_results)
print("The confusion matrix is:")
print(conf_mat)
print("\n\nOverall result:")
print(classification_report(target_test, predict_results))
