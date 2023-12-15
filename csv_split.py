import csv
import random


input_csv_path = '/media/hkuit164/Backup/xjl/20231207_kpsVideo/tinyv7_train/total.csv'
train_csv = '/media/hkuit164/Backup/xjl/20231207_kpsVideo/tinyv7_train/train.csv'
test_csv = '/media/hkuit164/Backup/xjl/20231207_kpsVideo/tinyv7_train/test.csv'
split_ratio = 0.7

with open(input_csv_path, 'r') as input_file:
    reader = csv.reader(input_file)
    lines = list(reader)

random.shuffle(lines)

split_index = int(len(lines) * split_ratio)

lines1 = lines[:split_index]
lines2 = lines[split_index:]

with open(train_csv, 'w', newline='') as output_file1:
    writer = csv.writer(output_file1)
    writer.writerows(lines1)

with open(test_csv, 'w', newline='') as output_file2:
    writer = csv.writer(output_file2)
    writer.writerows(lines2)
