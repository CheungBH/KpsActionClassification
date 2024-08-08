from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB

import csv
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump

def load_data_feature(file_path, feature_num):
    data = []
    traffic_feature = []
    csv_file = csv.reader(open(file_path))

    for content in csv_file:
        content = list(map(float, content[:-1]))
        if len(content) != 0:
            data.append(content)
            traffic_feature.append(content[0:feature_num])
    return traffic_feature

def load_data_target(file_path, feature_num):
    data = []
    traffic_target = []
    csv_file = csv.reader(open(file_path))

    for content in csv_file:
        content = list(map(float, content[:-1]))
        if len(content) != 0:
            data.append(content)
            traffic_target.append(content[feature_num])
    return traffic_target

def data_transform(traffic_feature):
    scaler = StandardScaler()
    scaler.fit(traffic_feature)
    traffic_feature = scaler.transform(traffic_feature)
    return traffic_feature

def train_model(model, feature_train, target_train, feature_test, target_test):
    model.fit(feature_train, target_train)
    predict_results = model.predict(feature_test)
    return predict_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--feature_num", type=int, default=26, help="Number of features in the dataset")
    parser.add_argument("--output_model_prefix", "-o", type=str, default="model", help="Prefix for output model filenames")
    args = parser.parse_args()

    models = [
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=3)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0)),
        ("LogisticRegression", LogisticRegression(solver='liblinear')),
        ("RandomForestClassifier", RandomForestClassifier()),
        ("AdaBoostClassifier", AdaBoostClassifier(n_estimators=1000)),
        ("SVM", svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')),
        ("BernoulliNB", BernoulliNB()),
        ("BaggingClassifier", BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=None), n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1))
    ]

    for model_name, model in models:
        if model_name in ["BaggingClassifier", "BernoulliNB", "SVM", "RandomForestClassifier"]:
            feature_tmp = load_data_feature(args.file_path, args.feature_num)
            feature = data_transform(feature_tmp)
            feature_train, feature_test, target_train, target_test = train_test_split(feature, load_data_target(args.file_path, args.feature_num), test_size=0.3, random_state=0)

            output_model_path = args.output_model_prefix + "_" + model_name + ".joblib"
            dump(predict_results, output_model_path)

        else:
            feature_train, feature_test, target_train, target_test = train_test_split(load_data_feature(args.file_path, args.feature_num), load_data_target(args.file_path, args.feature_num), test_size=0.3, random_state=0)
            predict_results = train_model(model, feature_train, target_train, feature_test, target_test)
            output_model_path = args.output_model_prefix + "_" + model_name + ".joblib"
            dump(predict_results, output_model_path)

        print("\nAlgorithm selected: ", model_name, "\n")
        print("Testing accuracy is {}\n".format(accuracy_score(predict_results, target_test)))
        conf_mat = confusion_matrix(target_test, predict_results)
        print("The confusion matrix is:")
        print(conf_mat)
        print("\n\nOverall result:")
        print(classification_report(target_test, predict_results))

if __name__ == "__main__":
    main()
