import numpy as np
import joblib
import csv
import time
import argparse

parser = argparse.ArgumentParser(description='ML Model Prediction')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--model_path', type=str, help='Path to the ML model (.joblib)')
args = parser.parse_args()

csv_file = csv.reader(open(args.csv_path))
ML_model = joblib.load(args.model_path)

total_time = 0

for i in csv_file:
    data = list(map(float, i[:-1]))
    content = [data[:-1]]
    label = [data[-1]]
    start_time = time.time()
    prediction = ML_model.predict(np.array(content))
    end_time = time.time()
    execution_time = end_time - start_time
    total_time += execution_time
    # print("Prediction:", prediction)
    # print("Label:", label)

print("Total execution time:", total_time, "s")
