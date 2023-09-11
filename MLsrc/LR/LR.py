# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
#
# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
# model.score(X_test, y_test)


from sklearn.linear_model import LogisticRegression
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

feature_num = 8
data=[]
traffic_feature=[]
traffic_target=[]

csv_file = csv.reader(open("../yoga_action.csv"))
for content in csv_file:
    content=list(map(float,content[:-1]))
    if len(content)!=0:
        data.append(content)
        traffic_feature.append(content[0:feature_num-1])
        traffic_target.append(content[feature_num])
# print('data=',data)
# print('traffic_feature=',traffic_feature)
# print('traffic_target=',traffic_target)

feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3,random_state=0)
LR = LogisticRegression()
LR.fit(feature_train,target_train)
predict_results=LR.predict(feature_test)

print("Testing accuracy is {}\n".format(accuracy_score(predict_results, target_test)))
conf_mat = confusion_matrix(target_test, predict_results)
print("The confusion matrix is:")
print(conf_mat)
print("\n\nOverall result:")
print(classification_report(target_test, predict_results))

