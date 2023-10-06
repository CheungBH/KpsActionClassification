import numpy as np
import joblib
import csv
import time
import argparse
import os

parser = argparse.ArgumentParser(description='ML Model Prediction')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--model_path', type=str, help='Path to the ML model (.joblib) or directory containing joblib files')
args = parser.parse_args()

# csv_file = csv.reader(open(args.csv_path))
models = {}
if os.path.isdir(args.model_path):
    model_files = [f for f in os.listdir(args.model_path) if f.endswith('.joblib')]
    for file in model_files:
        model_path = os.path.join(args.model_path, file)
        model_name = os.path.splitext(file)[0]
        models[model_name] = joblib.load(model_path)
else:
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    models[model_name] = joblib.load(args.model_path)

results = {}

csv_data = list(csv.reader(open(args.csv_path)))
# header = csv_data[0]
# data_rows = csv_data[1:]
for model_name, model in models.items():
    total_time = 0
    correct_predictions = 0
    total_samples = 0

    for row in csv_data:
        data = list(map(float, row[:-1]))
        content = [data[:-1]]
        label = data[-1]
        start_time = time.time()

        prediction = model.predict(np.array(content))
        if prediction == label:
            correct_predictions += 1
        total_samples += 1

        end_time = time.time()
        execution_time = end_time - start_time
        total_time += execution_time

    accuracy = (correct_predictions / total_samples) * 100
    results[model_name] = {
        'total_time': total_time,
        'accuracy': accuracy
    }
output_csv = "inference_results.csv"
print("Model\t\tTotal Time (s)\tAccuracy (%)")
for model_name, result in results.items():
    print(f"{model_name}\t{result['total_time']:.6f}\t\t{result['accuracy']:.6f}")
    with open(output_csv, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, result['total_time'], result['accuracy']])
