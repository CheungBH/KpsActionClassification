from sklearn import svm
import argparse
import csv
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="Path to the training CSV file")
    parser.add_argument("--test_file", type=str, help="Path to the testing CSV file")
    parser.add_argument("--config", type=str, help="Load the config file of the algorithm")
    parser.add_argument("--output_folder", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    feature_num = 26
    train_feature, train_target = load_data(args.train_file, feature_num)
    test_feature, test_target = load_data(args.test_file, feature_num)

    scaler = StandardScaler()
    scaler.fit(train_feature)
    train_feature = scaler.transform(train_feature)
    scaler.fit(test_feature)
    test_feature = scaler.transform(test_feature)

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    clf = svm.SVC(**config)
    clf.fit(train_feature, train_target)

    predict_results = clf.predict(test_feature)
    dump(clf, f"{args.output_folder}/model.joblib")

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

    csv_file_path = "../exp/results.csv"

    result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(cls_result))]
    result_data = ["SVM", train_accuracy, test_accuracy] + [acc for acc in cls_result]

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

if __name__ == "__main__":
    main()
