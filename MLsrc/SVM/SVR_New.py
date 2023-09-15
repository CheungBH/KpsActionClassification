from sklearn.svm import SVR
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


feature_num = 26
data = []
traffic_feature = []
traffic_target = []
csv_file = csv.reader(open("../tennis_player_4cls.csv"))

for content in csv_file:
    content = list(map(float, content[:-1]))
    if len(content) != 0:
        data.append(content)
        traffic_feature.append(content[0:feature_num-1])
        traffic_target.append(content[feature_num])

feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3, random_state=0)

svr = SVR()
svr.fit(feature_train, target_train)

predictions = svr.predict(feature_test)
print(predictions)

mse = mean_squared_error(target_test, predictions)
print("Mean Squared Error: ", mse)
