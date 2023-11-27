from sklearn import svm
import argparse
import csv
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

    scaler = StandardScaler()
    scaler.fit(train_feature)
    train_feature = scaler.transform(train_feature)
    scaler.fit(test_feature)
    test_feature = scaler.transform(test_feature)

    if save_model is True:
        clf = svm.SVC()
        clf.fit(train_feature, train_target)
        predict_results_train = clf.predict(train_feature)
        predict_results_test = clf.predict(test_feature)
        dump(clf, f"{args.output_folder}/model.joblib")

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

    else:
        config = []
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for rp in [0.1, 0.5, 1, 5, 10]: # C
                for shrink in [True, False]:
                    for prob in [True, False]:
                        for dfs in ["ovo", "ovr"]:
                            if kernel == "rbf" or kernel == "poly" or kernel == "sigmoid":
                                for gam in ["scale", "auto"]:
                                    for deg in [2, 3, 5]:
                                        if kernel == "poly":
                                            clf = svm.SVC(C=rp, kernel=kernel, shrinking=shrink, probability=prob, decision_function_shape=dfs, degree=deg, gamma=gam)
                                            config = [f"C: {rp}, kernel: {kernel}, shrinking: {shrink}, probability: {prob}, decision_function_shape: {dfs}, degree: {deg}, gamma: {gam}"]
                                        else:
                                            clf = svm.SVC(C=rp, kernel=kernel, shrinking=shrink, probability=prob, decision_function_shape=dfs, gamma=gam)
                                            config = [f"C: {rp}, kernel: {kernel}, shrinking: {shrink}, probability: {prob}, decision_function_shape: {dfs}, gamma: {gam}"]
                            else:
                                clf = svm.SVC(C=rp, kernel=kernel, shrinking=shrink, probability=prob, decision_function_shape=dfs)
                                config = [f"C: {rp}, kernel: {kernel}, shrinking: {shrink}, probability: {prob}, decision_function_shape: {dfs}"]

                            clf.fit(train_feature, train_target)
                            predict_results_train = clf.predict(train_feature)
                            predict_results_test = clf.predict(test_feature)

                            train_accuracy = accuracy_score(predict_results_train, train_target)
                            print("\n\nSVM" + f"{config}")
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
                            result_data = ["SVM", train_accuracy, test_accuracy] + [acc for acc in cls_result] + [f"{config}"]

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
