import csv
import json
import os
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
from augmentor import Augmentor


algorithm_dict = {
    'knn': KNeighborsClassifier,
    'GBDT': GradientBoostingClassifier,
    'DeTree': DecisionTreeClassifier,
    'LR': LogisticRegression,
    'RF': RandomForestClassifier,
    'AdaBoost': AdaBoostClassifier,
    'SVM': SVC,
    'Bayes': BernoulliNB,
    'bagging': BaggingClassifier
}

def load_data(file_path, feature_num, augment=False):
    data_feature = []
    data_target = []
    csv_file = csv.reader(open(file_path))
    for content in csv_file:
        content = list(map(float, content[:-1]))
        if len(content) != 0:
            data_feature.append(content[0:feature_num])
            data_target.append(content[feature_num])

    if augment is True:
        augmentor = Augmentor()
        augmentor.augment(data_feature, data_target)
        data_feature, data_target = augmentor.append_dataset(data_feature, data_target)
    return data_feature, data_target


def data_transform(traffic_feature):
    scaler = StandardScaler()
    scaler.fit(traffic_feature)
    traffic_feature = scaler.transform(traffic_feature)
    return traffic_feature


def run_algorithm(config_file, train_file, test_file, config_folder, output_folder, csv_file_path, box_size, box_ratio, feature_num, save_score):

    if box_size is True:
        feature_num += 8

    if box_ratio is True:
        feature_num += 1
    train_feature, train_target = load_data(train_file, feature_num, augment=True)
    test_feature, test_target = load_data(test_file, feature_num)

    # config_file = os.path.join(config_folder, f"{algorithm_name}_cfg.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    algorithm_name = config["name"]
    config.pop("name")

    if algorithm_name not in algorithm_dict:
        raise ValueError("Unsupported algorithm")

    if algorithm_name in ['RF', 'SVM', 'Bayes', 'bagging']:
        train_feature = data_transform(train_feature)
        test_feature = data_transform(test_feature)

    algorithm_class = algorithm_dict[algorithm_name]
    algorithm = algorithm_class(**config)
    algorithm.fit(train_feature, train_target)
    predict_results_train = algorithm.predict(train_feature)
    predict_results_test = algorithm.predict(test_feature)
    test_accuracy = accuracy_score(predict_results_test, test_target)
    if (test_accuracy * 100) > save_score:
        config_name = config_file.split("/")[-1].split(".")[0]
        dump(algorithm, f"{output_folder}/{config_name}_model.joblib")
    else:
        for save_idx in range(10):
            algorithm.fit(train_feature, train_target)
            predict_results_test = algorithm.predict(test_feature)
            test_accuracy = accuracy_score(predict_results_test, test_target)
            if (test_accuracy * 100) > save_score:
                dump(algorithm, f"{output_folder}/{algorithm_name}_model.joblib")
                break


    train_accuracy = accuracy_score(predict_results_train, train_target)
    print(f"Training accuracy for {algorithm_name} is: {train_accuracy}")
    print(f"Testing accuracy for {algorithm_name} is: {test_accuracy}")

    print("\nThe train confusion matrix is:")
    train_conf_mat = confusion_matrix(train_target, predict_results_train)
    print(train_conf_mat)

    train_cls_result = []
    train_id = 0
    for row in train_conf_mat:
        row_sum = sum(row)
        row_result = row[train_id] / row_sum
        train_cls_result.append(row_result)
        train_id += 1

    print("\nThe test confusion matrix is:")
    test_conf_mat = confusion_matrix(test_target, predict_results_test)
    print(test_conf_mat)

    test_cls_result = []
    test_id = 0
    for row in test_conf_mat:
        row_sum = sum(row)
        row_result = row[test_id] / row_sum
        test_cls_result.append(row_result)
        test_id += 1

    print("\nTrain Class accuracy: ")
    print(train_cls_result)
    print("\nTest Class accuracy: ")
    print(test_cls_result)

    print("\n\nOverall train result:")
    print(classification_report(train_target, predict_results_train))
    print("\n\nOverall test result:")
    print(classification_report(test_target, predict_results_test))

    # csv_file_path = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/csv/results.csv"

    result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(test_cls_result))]
    result_data = [algorithm_name, train_accuracy, test_accuracy] + [acc for acc in test_cls_result]

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
            # writer.writerow(test_conf_mat)
        else:
            writer.writerow(result_data)
            # writer.writerow(test_conf_mat)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_file", type=str, help="Path to the training CSV file", default="data/20231207_ML_model/train.csv")
    # parser.add_argument("--test_file", type=str, help="Path to the testing CSV file", default="data/20231207_ML_model/test.csv")
    parser.add_argument("--data_folder", type=str, help="Path to the folder of CSV files", required=True)
    parser.add_argument("--config_folder", type=str, help="Path to the folder containing config files", default="cfg")
    parser.add_argument("--output_folder", type=str, help="Path to save the trained models", default="exp")
    parser.add_argument("--feature_number", type=int, default=34)
    parser.add_argument("--save_score", type=int, default=0)

    args = parser.parse_args()
    train_file, test_file = os.path.join(args.data_folder, "train.csv"), os.path.join(args.data_folder, "val.csv")

    feature_number = args.feature_number
    os.makedirs(args.output_folder, exist_ok=True)
    save_score = args.save_score
    bbox_size = False
    bbox_hw_ratio = False
    algorithm_configs = [os.path.join(args.config_folder, config) for config in os.listdir(args.config_folder)]
    # algorithm_names = ['knn', 'GBDT', 'DeTree', 'LR', 'RF', 'AdaBoost', 'SVM', 'Bayes', 'bagging']
    output_csv = os.path.join(args.output_folder, "results.csv")
    for idx, algorithm_config in enumerate(algorithm_configs):
        print(f"Running {algorithm_config}")
        run_algorithm(algorithm_config, train_file, test_file, args.config_folder, args.output_folder,
                      output_csv, bbox_size, bbox_hw_ratio, feature_number, save_score)


if __name__ == "__main__":
    main()
