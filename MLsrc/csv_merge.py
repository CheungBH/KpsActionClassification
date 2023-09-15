import argparse
import csv
import os

parser = argparse.ArgumentParser(description='Merge CSV files in a folder.')
parser.add_argument('folder', help='Folder path containing CSV files')
parser.add_argument('output', help='Output file name')

args = parser.parse_args()

folder_path = args.folder
output_file = args.output

csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, 'r') as infile:
            reader = csv.reader(infile)
            for row in reader:
                writer.writerow(row)
