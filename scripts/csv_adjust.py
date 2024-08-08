import csv


def update_csv(file_path, output_file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    for i in range(len(data)):
        if data[i][34] == "0" or data[i][34] == "1":
            data[i][34] = "0"
        elif data[i][34] == "2":
            data[i][34] = "1"

    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


input_file_path = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/1_final_data/ml_3cls(bfo)/val.csv'
output_file_path = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/1_final_data/ml_2cls(overhead)/val.csv'
update_csv(input_file_path, output_file_path)