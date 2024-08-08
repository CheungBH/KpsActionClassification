import csv
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump
import argparse

def load_data(file_path, feature_num):
    data_feature = []
    data_target = []
    csv_file = csv.reader(open(file_path))
    for content in csv_file:
        content = list(map(float, content[:-1]))
        if len(content) != 0:
            data_feature.append(content[0:feature_num])
            data_target.append(content[feature_num])
    return data_feature, data_target

def data_transform(traffic_feature):
    scaler = StandardScaler()
    scaler.fit(traffic_feature)
    traffic_feature = scaler.transform(traffic_feature)
    return traffic_feature

def run_algorithm(algorithm_name, train_file, test_file, config_file, output_folder):
    feature_num = 26
    train_feature, train_target = load_data(train_file, feature_num)
    test_feature, test_target = load_data(test_file, feature_num)

    with open(config_file, 'r') as config_file:
        config = json.load(config_file)

    algorithm_dict = {
        'knn': KNeighborsClassifier,
        'GBDT': GradientBoostingClassifier,
        'DeTree': DecisionTreeClassifier,
        'LR': LogisticRegression,
        'RF': RandomForestClassifier,
        'Adaboost': AdaBoostClassifier,
        'SVM': SVC,
        'Bayes': BernoulliNB,
        'bagging': BaggingClassifier
    }

    if algorithm_name not in algorithm_dict:
        raise ValueError("Unsupported algorithm")

    algorithm_class = algorithm_dict[algorithm_name]
    algorithm = algorithm_class(**config)

    if algorithm_name in ['RF', 'SVM', 'Bayes', 'bagging']:
        train_feature = data_transform(train_feature)
        test_feature = data_transform(test_feature)

    algorithm.fit(train_feature, train_target)
    predict_results = algorithm.predict(test_feature)
    dump(algorithm, f"{output_folder}/{algorithm_name}_model.joblib")

    train_accuracy = accuracy_score(predict_results, train_target)
    print("Training accuracy is : ", train_accuracy)
    test_accuracy = accuracy_score(predict_results, test_target)
    print("\nTesting accuracy is : ", test_accuracy)
    print("\nThe confusion matrix is:")
    conf_mat = confusion_matrix(test_target, predict_results)
    print(conf_mat)
    cls_result = []
    id = 0
    for row in conf_mat:
        row_sum = sum(row)
        row_result = row[id] / row_sum
        cls_result.append(row_result)
        id += 1

    print("\nClass accuracy: ")
    print(cls_result)
    print("\n\nOverall result:")
    print(classification_report(test_target, predict_results))

    csv_file_path = "results.csv"

    result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(cls_result))]
    result_data = [algorithm_name, train_accuracy, test_accuracy] + [acc for acc in cls_result]

    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        empty = True
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row:
                    empty = False
        if empty:
            writer.writerow(result_title)
            writer.writerow(result_data)
        else:
            writer.writerow(result_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, help="Name of the algorithm")
    # knn, GBDT, Bayes, bagging, Adaboost, SVM, DeTree, RF, LR
    parser.add_argument("--train_file", type=str, help="Path to the training CSV file")
    parser.add_argument("--test_file", type=str, help="Path to the testing CSV file")
    parser.add_argument("--config", type=str, help="Load the algorithm with its config file")
    parser.add_argument("--output_folder", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    run_algorithm(args.algorithm, args.train_file, args.test_file, args.config, args.output_folder)

if __name__ == "__main__":
    main()
