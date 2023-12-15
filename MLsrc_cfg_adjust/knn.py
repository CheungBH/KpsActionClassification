from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="Path to the training CSV file")
    parser.add_argument("--test_file", type=str, help="Path to the testing CSV file")
    parser.add_argument("--output_folder", type=str, help="Path to save the trained model")
    parser.add_argument("--csv_path", type=str, help="Path to save the results in csv")
    args = parser.parse_args()

    save_model = False

    feature_num = 34
    train_feature, train_target = load_data(args.train_file, feature_num)
    test_feature, test_target = load_data(args.test_file, feature_num)

    if save_model is True:
        knn = KNeighborsClassifier()
        knn.fit(train_feature, train_target)
        predict_results_train = knn.predict(train_feature)
        predict_results_test = knn.predict(test_feature)
        dump(knn, f"{args.output_folder}/model.joblib")

        train_accuracy = accuracy_score(predict_results_train, train_target)
        print("Training accuracy is : ", train_accuracy)
        test_accuracy = accuracy_score(predict_results_test, test_target)
        print("\nTesting accuracy is : ", test_accuracy)
        print("\nThe confusion matrix is:")
        conf_mat = confusion_matrix(test_target, predict_results_test)
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
        print(classification_report(test_target, predict_results_test))

        csv_file_path = "../exp/results.csv"

        result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(cls_result))]
        result_data = ["knn", train_accuracy, test_accuracy] + [acc for acc in cls_result]

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
    else:
        for neighbor in [3, 5, 7, 9, 11]:
            for weight in ["uniform", "distance"]:
                for power in [1, 2, 3, 10]:
                    knn = KNeighborsClassifier(n_neighbors=neighbor, weights=weight, p=power)
                    knn.fit(train_feature, train_target)
                    predict_results_train = knn.predict(train_feature)
                    predict_results_test = knn.predict(test_feature)

                    train_accuracy = accuracy_score(predict_results_train, train_target)
                    print("\n\nKNN " + f"Neighbors: {neighbor} " + f"Weights: {weight} " + f"P: {power}")

                    print("Training accuracy is : ", train_accuracy)
                    test_accuracy = accuracy_score(predict_results_test, test_target)
                    print("Testing accuracy is : ", test_accuracy)
                    print("The test confusion matrix is:")
                    conf_mat = confusion_matrix(test_target, predict_results_test)
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
                    print(classification_report(test_target, predict_results_test))
                    csv_file_path = args.csv_path

                    result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(cls_result))]
                    result_data = ["knn", train_accuracy, test_accuracy] + [acc for acc in cls_result] + [f"Neighbors: {neighbor} " + f"Weights: {weight} " + f"P: {power}"]

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
