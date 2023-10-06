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


def train_knn_model(feature_train, target_train, feature_test, target_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(feature_train, target_train)
    predict_results = knn.predict(feature_test)
    return predict_results

def train_gbdt_model(feature_train, target_train, feature_test, target_test):
    gbdt = GradientBoostingClassifier(random_state=0)
    gbdt.fit(feature_train, target_train)
    predict_results = gbdt.predict(feature_test)
    return predict_results

def train_de_tree_model(feature_train, target_train, feature_test, target_test):
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(feature_train, target_train)
    predict_results = dt.predict(feature_test)
    return predict_results

def train_LR_model(feature_train, target_train, feature_test, target_test):
    LR = LogisticRegression(solver='liblinear')
    LR.fit(feature_train, target_train)
    predict_results = LR.predict(feature_test)
    return predict_results

def train_RF_model(feature_train, target_train, feature_test, target_test):
    clf = RandomForestClassifier()
    clf.fit(feature_train, target_train)
    predict_results = clf.predict(feature_test)
    return predict_results

def train_AdaBoost_model(feature_train, target_train, feature_test, target_test):
    AB = AdaBoostClassifier(n_estimators=1000)
    AB.fit(feature_train, target_train)
    predict_results = AB.predict(feature_test)
    return predict_results

def train_SVM_model(feature_train, target_train, feature_test, target_test):
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    # clf = svm.SVC(C=10, kernel='poly', decision_function_shape='ovr')
    clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
    clf.fit(feature_train,target_train)
    predict_results = clf.predict(feature_test)
    return predict_results

def train_bayes_model(feature_train, target_train, feature_test, target_test):
    NB = BernoulliNB()
    NB.fit(feature_train, target_train)
    predict_results = NB.predict(feature_test)
    return predict_results

def train_bagging_model(feature_train, target_train, feature_test, target_test):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
    clf.fit(feature_train, target_train)
    predict_results = clf.predict(feature_test)
    return predict_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", "-a", type=str,
                        choices=["knn", "gbdt", "de_tree", "LR", "RF", "AdaBoost", "SVM", "bayes", "bagging"],
                        help="Choose the algorithm to train")
    parser.add_argument("--file_path", type=str, help="Path to the CSV file")
    parser.add_argument("--feature_num", type=int, default=26, help="Number of features in the dataset")
    parser.add_argument("--output_model", "-o", type=str, help="Path to model -- .joblib")
    args = parser.parse_args()

    algorithm_mapping = {
        "knn": train_knn_model,
        "gbdt": train_gbdt_model,
        "de_tree": train_de_tree_model,
        "LR": train_LR_model,
        "RF": train_RF_model,
        "AdaBoost": train_AdaBoost_model,
        "SVM": train_SVM_model,
        "bayes": train_bayes_model,
        "bagging": train_bagging_model
    }

    if args.algorithm in algorithm_mapping:
        train_model_func = algorithm_mapping[args.algorithm]
    else:
        raise ValueError("Invalid algorithm choice.")


    if args.algorithm in ["knn", "gbdt", "de_tree", "LR", "AdaBoost"]:
        feature_train, feature_test, target_train, target_test = train_test_split(load_data_feature(args.file_path, args.feature_num), load_data_target(args.file_path, args.feature_num), test_size=0.3, random_state=0)

    else: # bagging SVM Bayes RF
        feature_tmp = load_data_feature(args.file_path, args.feature_num)
        feature = data_transform(feature_tmp)
        feature_train, feature_test, target_train, target_test = train_test_split(feature, load_data_target(args.file_path, args.feature_num), test_size=0.3, random_state=0)


    predict_results = train_model_func(feature_train, target_train, feature_test, target_test)
    dump(predict_results, args.output_model)

    print("\nAlgorithm selected: ", args.algorithm, "\n")
    print("Testing accuracy is {}\n".format(accuracy_score(predict_results, target_test)))
    conf_mat = confusion_matrix(target_test, predict_results)
    print("The confusion matrix is:")
    print(conf_mat)
    print("\n\nOverall result:")
    print(classification_report(target_test, predict_results))


if __name__ == "__main__":
    main()
