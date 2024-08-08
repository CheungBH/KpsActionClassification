import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import csv
import time

csv_file = csv.reader(open("../tennis_player_4cls.csv"))
ML_model = joblib.load('../../exp/Decision_Tree/dt_test.joblib')
total_time = 0
predictions = []
labels = []

for i in csv_file:
    data = list(map(float,i[:-1]))
    content = [data[:-1]]
    label = [data[-1]]
    start_time = time.time()
    prediction = ML_model.predict(np.array(content))
    end_time = time.time()
    execution_time = end_time - start_time
    total_time = total_time + execution_time

    predictions.append(prediction)
    labels.append(label)
    # print(prediction)
    # print(label)

accuracy = accuracy_score(labels, predictions)
print("Accuracy: ", accuracy)
print("Total execution time: ", total_time, "s")
