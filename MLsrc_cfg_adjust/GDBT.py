from sklearn.ensemble import GradientBoostingClassifier
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump
import argparse


def load_data(file_path, feature_num):
    data_feature = []
    data_target = []
    csv_file = csv.reader(open(file_path))
    for content in csv_file:
        content = list(map(float, content[:-2]))
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

    feature_num = 26
    train_feature, train_target = load_data(args.train_file, feature_num)
    test_feature, test_target = load_data(args.test_file, feature_num)

    if save_model is True:
        gbdt = GradientBoostingClassifier()
        gbdt.fit(train_feature, train_target)
        predict_results_train = gbdt.predict(train_feature)
        predict_results_test = gbdt.predict(test_feature)
        dump(gbdt, f"{args.output_folder}/model.joblib")

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

        csv_file_path = args.csv_path

        result_title = ["algo", 'Train Acc', 'Test Acc'] + ['Class ' + str(i) + ' Acc' for i in range(len(cls_result))]
        result_data = ["GBDT", train_accuracy, test_accuracy] + [acc for acc in cls_result]

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
        for lr in [0.1, 0.2, 0.3]:
            for est in [100, 200, 300]:
                for sub in [0.5, 1.0]:
                    for sample_split in [0.5, 2, 5]:
                        for sample_leaf in [0.5, 1, 5]:
                            for weight_fraction_leaf in [0, 0.1, 0.3]:
                                for depth in [1, 3, 10]:
                                    for impurity_decrease in [0, 0.5, 1]:
                                        gbdt = GradientBoostingClassifier(learning_rate=lr, n_estimators=est, subsample=sub, min_samples_split=sample_split, min_samples_leaf=sample_leaf, min_weight_fraction_leaf=weight_fraction_leaf, max_depth=depth, min_impurity_decrease=impurity_decrease)
                                        gbdt.fit(train_feature, train_target)
                                        predict_results_train = gbdt.predict(train_feature)
                                        predict_results_test = gbdt.predict(test_feature)

                                        train_accuracy = accuracy_score(predict_results_train, train_target)
                                        # print("\n\nGBDT " + f"Learning_rate: {lr}, n_estimators: {est}, subsample: {sub}, min_samples_split: {sample_split}, min_samples_leaf: {sample_leaf}, min_weight_fraction_leaf: {weight_fraction_leaf}, max_depth: {depth}, min_impurity_decrease: {impurity_decrease}")
                                        # print("Training accuracy is : ", train_accuracy)
                                        test_accuracy = accuracy_score(predict_results_test, test_target)
                                        # print("Testing accuracy is : ", test_accuracy)
                                        # print("The test confusion matrix is:")
                                        conf_mat = confusion_matrix(test_target, predict_results_test)
                                        # print(conf_mat)

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
                                        result_data = ["GBDT", train_accuracy, test_accuracy] + [acc for acc in cls_result] + [f"Learning_rate: {lr}, n_estimators: {est}, subsample: {sub}, min_samples_split: {sample_split}, min_samples_leaf: {sample_leaf}, min_weight_fraction_leaf: {weight_fraction_leaf}, max_depth: {depth}, min_impurity_decrease: {impurity_decrease}"]

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
