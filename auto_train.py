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
from MLsrc_range import generate_param_dict


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


def run_algorithm(algorithm_name, train_file, test_file, exp_id, root_folder, feature_num, save_score):

    csv_file_path = os.path.join(root_folder, "results.csv")


    name = "{}-{}".format(exp_id, algorithm_name)
    sub_folder = os.path.join(root_folder, name)
    os.makedirs(sub_folder, exist_ok=True)

    train_feature, train_target = load_data(train_file, feature_num, augment=True)
    test_feature, test_target = load_data(test_file, feature_num)


    if algorithm_name not in algorithm_dict:
        raise ValueError("Unsupported algorithm")

    if algorithm_name in ['RF', 'SVM', 'Bayes', 'bagging']:
        train_feature = data_transform(train_feature)
        test_feature = data_transform(test_feature)

    config = generate_param_dict(algorithm_name)
    algorithm = algorithm_dict[algorithm_name](**config)
    algorithm.fit(train_feature, train_target)
    config["algo"] = algorithm_name

    predict_results_train = algorithm.predict(train_feature)
    predict_results_test = algorithm.predict(test_feature)
    # if (test_accuracy * 100) > save_score:
        # config_name = config_file.split("/")[-1].split(".")[0]
    dump(algorithm, f"{sub_folder}/model.joblib")
    with open(f"{sub_folder}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    r_file = open(f"{sub_folder}/result.txt", "w")

    train_accuracy = accuracy_score(predict_results_train, train_target)
    print(f"Training accuracy for {algorithm_name} is: {train_accuracy}")
    test_accuracy = accuracy_score(predict_results_test, test_target)
    print(f"Testing accuracy for {algorithm_name} is: {test_accuracy}")

    if test_accuracy * 100 < save_score:
        return

    print("\nThe train confusion matrix is:")
    train_conf_mat = confusion_matrix(train_target, predict_results_train)
    print(train_conf_mat)
    r_file.write("Train confusion matrix\n")


    train_cls_result = []
    train_id = 0
    for row in train_conf_mat:
        row_sum = sum(row)
        row_result = row[train_id] / row_sum
        train_cls_result.append(row_result)
        r_file.write(str(row.tolist()) + "\n")
        train_id += 1

    print("\nThe test confusion matrix is:")
    test_conf_mat = confusion_matrix(test_target, predict_results_test)
    print(test_conf_mat)
    r_file.write("\nTest confusion matrix\n")
    # r_file.write(test_conf_mat)

    test_cls_result = []
    test_id = 0
    for row in test_conf_mat:
        row_sum = sum(row)
        row_result = row[test_id] / row_sum
        test_cls_result.append(row_result)
        r_file.write(str(row.tolist()) + "\n")
        test_id += 1

    print("\nTrain Class accuracy: ")
    print(train_cls_result)
    print("\nTest Class accuracy: ")
    print(test_cls_result)

    print("\n\nOverall train result:")
    train_report = classification_report(train_target, predict_results_train)
    print(train_report)
    r_file.write("\nTrain Report\n")
    r_file.write(train_report)

    print("\n\nOverall test result:")
    test_report = classification_report(test_target, predict_results_test)
    print(test_report)
    r_file.write("\nTest Report\n")
    r_file.write(test_report)

    result_title = ["algo", 'idx', 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(test_cls_result))]
    result_data = [algorithm_name, exp_id, train_accuracy, test_accuracy] + [acc for acc in test_cls_result]

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
    parser.add_argument("--data_folder", type=str, help="Path to the folder of CSV files", required=True)
    parser.add_argument("--output_folder", type=str, help="Path to save the trained models", default="exp/random")
    parser.add_argument("--feature_number", type=int, default=34)
    parser.add_argument("--save_score", type=int, default=0)
    parser.add_argument("--algo", type=str, default="all")
    parser.add_argument("--repeat_time", type=int, default=20)

    args = parser.parse_args()
    train_file, test_file = os.path.join(args.data_folder, "train.csv"), os.path.join(args.data_folder, "val.csv")

    feature_number = args.feature_number
    os.makedirs(args.output_folder, exist_ok=True)
    save_score = args.save_score

    algorithms = [args.algo] if args.algo != "all" else list(algorithm_dict.keys())
    for algorithm in algorithms:
        for idx in range(args.repeat_time):
            run_algorithm(algorithm, train_file, test_file, idx, args.output_folder, feature_number, save_score)



if __name__ == "__main__":
    main()
