import numpy as np
import joblib
import csv
import time
import argparse
import os
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='ML Model Prediction')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--model_path', type=str, help='Path to the ML model (.joblib)')
parser.add_argument('--output_path', type=str, default="inf_vis.csv", help='Path to the CSV file')
args = parser.parse_args()

csv_file = csv.reader(open(args.csv_path))
model_name = os.path.splitext(os.path.basename(args.model_path))[0]
ML_model = joblib.load(args.model_path)
output_csv = args.output_path

total_time = 0
predictions = []
labels = []
img_names = []

for i in csv_file:
    data = list(map(float, i[:-2]))
    content = [data[:-1]]
    label = data[-1]
    img_name = [i[-1]]
    start_time = time.time()
    prediction = ML_model.predict(np.array(content))
    end_time = time.time()
    execution_time = end_time - start_time
    total_time += execution_time

    predictions.append(int(prediction))
    labels.append(int(label))
    img_names.append(img_name)
accuracy = accuracy_score(labels, predictions)

for i in range(len(predictions)):
    print("Prediction: ", predictions[i], " | Actual Label: ", labels[i])

print("Accuracy: ", accuracy)
print("Total execution time:", total_time, "s")

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([model_name])
    writer.writerow(['', 'Labels', 'Predictions'])

    if len(img_names) == len(labels) == len(predictions):
        for img_name, label, prediction in zip(img_names, labels, predictions):
            writer.writerow([img_name, label, prediction])
    else:
        print("Error: img_names, labels and predictions must have the same length")
