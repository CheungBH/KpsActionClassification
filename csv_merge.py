import argparse
import csv
import os

parser = argparse.ArgumentParser(description='Merge CSV files in a folder.')
parser.add_argument('--input_folder', '-i', type=str, help='Folder path containing CSV files')
parser.add_argument('--output_file', '-o', type=str, help='Output file name')
args = parser.parse_args()

folder_path = args.input_folder
output_file = args.output_file

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

def sort_by_column(data, column_index):
    sorted_data = sorted(data, key=lambda row: float(row[column_index]), reverse=True)
    return sorted_data

with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                writer.writerow(row)

