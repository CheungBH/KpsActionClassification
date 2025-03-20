import json, csv
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from joblib import dump
import argparse
from MLsrc_range_regression import generate_param_dict

algorithm_dict = {
    'SVR': SVR,
    'Tree': DecisionTreeRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'MLPRegressor': MLPRegressor
}


def load_data(file_path, feature_num):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :feature_num]  # First three columns as features
    y = data.iloc[:, feature_num:]  # Fourth column as target
    return X, y


def run_algorithm(algorithm_name, train_file, test_file, exp_id, root_folder, feature_num=2):
    csv_file_path = os.path.join(root_folder, "results.csv")

    name = "{}-{}".format(exp_id, algorithm_name)
    sub_folder = os.path.join(root_folder, name)
    os.makedirs(sub_folder, exist_ok=True)

    X_train, y_train = load_data(train_file, feature_num)
    X_test, y_test = load_data(test_file, feature_num)

    if algorithm_name not in algorithm_dict:
        raise ValueError("Unsupported algorithm")

    config = generate_param_dict(algorithm_name)
    algorithm = algorithm_dict[algorithm_name](**config)

    algorithm.fit(X_train, y_train)
    predictions = algorithm.predict(X_test)

    dump(algorithm, f"{sub_folder}/model.joblib")
    mse = mean_squared_error(y_test, predictions)
    print(f"{algorithm_name} model is saved as '{sub_folder}/model.joblib' with MSE: {mse}")
    mse_train = mean_squared_error(y_train, algorithm.predict(X_train))

    with open(f"{sub_folder}/result.txt", "w") as r_file:
        r_file.write(f"Test Mean Squared Error: {mse}\n")
        r_file.write(f"Train Mean Squared Error: {mse_train}\n")

    result_title = ["algo", 'idx', 'MSE_train', "MSE_test"]
    result_data = [algorithm_name, exp_id, mse_train, mse]
    with open(f"{sub_folder}/config.json", "w") as f:
        json.dump(config, f, indent=4)


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
    parser.add_argument("--output_folder", type=str, help="Path to save the trained models", default="exp_reg")
    parser.add_argument("--algo", type=str, default="all")
    parser.add_argument("--repeat_time", type=int, default=20)
    parser.add_argument("--feature_number", type=int, default=-1)

    args = parser.parse_args()
    train_file, test_file = os.path.join(args.data_folder, "train.csv"), os.path.join(args.data_folder, "val.csv")

    os.makedirs(args.output_folder, exist_ok=True)

    algorithms = [args.algo] if args.algo != "all" else list(algorithm_dict.keys())
    for algorithm in algorithms:
        for idx in range(args.repeat_time):
            run_algorithm(algorithm, train_file, test_file, idx, args.output_folder, args.feature_number)


if __name__ == '__main__':
    main()
