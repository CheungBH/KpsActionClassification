from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump

feature_num = 26
data=[]
traffic_feature=[]
traffic_target=[]
csv_file = csv.reader(open("../tennis_player_4cls.csv"))

for content in csv_file:
    content = list(map(float, content[:-1]))
    if len(content) != 0:
        data.append(content)
        traffic_feature.append(content[0:feature_num])
        traffic_target.append(content[feature_num])

feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.27, random_state=0)

dt = DecisionTreeClassifier(random_state=0)
dt.fit(feature_train, target_train)

predict_results = dt.predict(feature_test)
dump(dt, '../exp/Decision_Tree/test/dt_test.joblib')

print("Testing accuracy is {}\n".format(accuracy_score(predict_results, target_test)))
conf_mat = confusion_matrix(target_test, predict_results)
print("The confusion matrix is:")
print(conf_mat)
print("\n\nOverall result:")
print(classification_report(target_test, predict_results))
