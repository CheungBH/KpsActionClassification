import json
import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
import numpy as np
import pandas as pd

prev_classes_list = ["left", "right", "overhead", "waiting"] # Add your desired classes here
novel_classes_list = ["not sure"]
classes_list = prev_classes_list + novel_classes_list

image_size = 300
input_type = "json"
input_csv_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_4cls_lr_0726/33.csv"
input_json_path = "/Users/cheungbh/Downloads/result1.json"
output_csv_path = "/media/hkuit164/Backup/KpsActionClassification/test.csv"
# choose how many frame merge togerther
merge_frame_size = 5
# choose the merge step
merge_frame_step = 5
frame_step = 5
kps_size = 34
label_seq = kps_size*merge_frame_size+1

# output_csv_path = "/media/hkuit164/Backup/2.csv"
# folder_path = "/media/hkuit164/Backup/yolov7ps/yolov7/runs/detect/exp19/crops/person"  # Specify the crop image folder path directly
# target_file = "classification_results.txt"  # Specify the target file name directly

def read_json_data(json_path):
    id1_data, id2_data = [], []
    with open(json_path, "r") as f:
        content = f.read()
        data_content = json.loads(content)
        for key, value in data_content.items():
            if '1.0' in value.keys():
                norm_kps_1 = normalize_keypoints(value['1.0']["kp"], value['1.0']['box'])
            if '2.0' in value.keys():
                norm_kps_2 = normalize_keypoints(value['2.0']["kp"], value['2.0']['box'])
            id1_frame_data = []
            id2_frame_data = []
            for position in norm_kps_1:
                for xy_coordinate in position:
                    id1_frame_data.append(xy_coordinate)

            for position in norm_kps_2:
                for xy_coordinate in position:
                    id2_frame_data.append(xy_coordinate)


            id1_data.append(id1_frame_data)
            id2_data.append(id2_frame_data)

        id1_merge_list = merge_sublists(id1_data, merge_frame_size, merge_frame_step, frame_step)
        id2_merge_list = merge_sublists(id2_data, merge_frame_size, merge_frame_step, frame_step)

    return id1_merge_list, id2_merge_list

def normalize_keypoints(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    normalized_keypoints = []

    for (x_kp, y_kp) in keypoints:
        x_norm = (x_kp - x1) / width
        y_norm = (y_kp - y1) / height
        normalized_keypoints.append([x_norm, y_norm])

    return normalized_keypoints

def merge_sublists(nest_list, merge_frame_size, merge_step, frame_step):
    merged_list = []
    for i in range(0, len(nest_list) - merge_frame_size - frame_step * (merge_frame_size - 1), merge_step):
        # for
        merge_sublist = []
        for j in range(merge_frame_size):
            merge_sublist += nest_list[i+j*frame_step]

        # merge_sublist = [item for sublist in nest_list[i:i+merge_frame_size] for item in sublist]
        # add default class
        merge_sublist.append(0)
        # add file name
        merge_sublist.append(input_json_path)
        merged_list.append(merge_sublist)

    return merged_list


class ImageLabellerGUI:
    def __init__(self, root, input_type):
        self.root = root
        self.image_files = []
        # down_train csv
        # self.current_index = 1112
        # down_val csv
        # self.current_index = 254
        self.current_index = 0
        self.classification_results = {}
        self.classes = []
        if input_type == "csv":
            data = pd.read_csv(input_csv_path)
            self.data = data.values.tolist()
        elif input_type == "json":
            self.data1, self.data2 = read_json_data(input_json_path)
            self.data = self.data1 + self.data2

        self.classes = classes_list  # Add your desired classes here
        self.create_widgets()
        self.load_data()
        self.update_text_view()
        # self.read_results()

    def create_widgets(self):
        self.root.title("Image Classifier")

        # Create buttons frame at the top
        self.buttons_frame_top = tk.Frame(self.root)
        self.buttons_frame_top.pack(side=tk.TOP, pady=10)

        # Create back button
        self.back_button = tk.Button(self.buttons_frame_top, text="Back", command=self.previous_image)
        self.back_button.pack(side=tk.LEFT, padx=5)
        self.back_button.config(state=tk.DISABLED)

        # Create next button
        self.next_button = tk.Button(self.buttons_frame_top, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        # Create a frame to display the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=10)

        # Create the text view
        self.text_view_frame = tk.Frame(self.root)
        self.text_view_frame.pack(pady=10)

        self.text_view_label = tk.Label(self.text_view_frame, text="Label:")
        self.text_view_label.pack(side=tk.LEFT)

        self.selected_image_label = tk.Label(self.text_view_frame, text="")
        self.selected_image_label.pack(side=tk.LEFT)

        # Create buttons frame at the bottom
        self.buttons_frame_bottom = tk.Frame(self.root)
        self.buttons_frame_bottom.pack(pady=10)

        # Create the counter label
        self.counter_label = tk.Label(self.buttons_frame_top, text="")
        self.counter_label.pack(side=tk.LEFT, padx=5)

        # Update the counter label
        self.update_counter_label()

        self.finish_button = tk.Button(self.root, text="Finish", command=self.show_warning)
        self.finish_button.pack(side=tk.TOP, pady=10)

        self.color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
        self.color_idx = [4, 7, 10, 13, 16]
        self.augment_factor = []
        self.connection = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (2, 4), (3, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]

    def choose_color(self, coord_i):
        c_idx = 0
        while True:
            if coord_i <= self.color_idx[c_idx]:
                return self.color_dict[c_idx]
            else:
                c_idx += 1

    def update_counter_label(self):
        total_data = len(self.data)
        current_data_number = self.current_index + 1
        self.counter_label.config(text=f"{current_data_number}/{total_data}")

    def load_data(self):

        self.class_buttons = []
        for i, class_name in enumerate(self.classes):
            button = tk.Button(self.buttons_frame_bottom, text=class_name, command=lambda x=class_name: self.label_image(x))
            button.grid(row=0, column=i, padx=5)
            self.class_buttons.append(button)

        self.display_image()
        self.update_button_colors()

    def display_image(self):
        sample = self.data[self.current_index]
        # def visualize(self, data=None, label=None):
        # for coord in data:
        imgs = []
        for i in range(merge_frame_size):
            data = sample[i*kps_size:(i+1)*kps_size]
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            float_single_coord = [x * image_size for x in data]
            for i in range(17):
                x = int(float_single_coord[i * 2])
                y = int(float_single_coord[i * 2 + 1])
                cv2.circle(image, (x, y), 5, self.choose_color(i), -1)
            # cv2.imshow("fdasjdas", image)
            # cv2.waitKey(0)
            for i, j in self.connection:
                x1, y1 = int(float_single_coord[i * 2]), int(float_single_coord[i * 2 + 1])
                x2, y2 = int(float_single_coord[j * 2]), int(float_single_coord[j * 2 + 1])
                cv2.line(image, (x1, y1), (x2, y2), self.choose_color(i), 2)
            imgs.append(image)
        # for img in imgs:

        merge_image = np.concatenate(imgs, axis=1)

        self.photo = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGBA)
        self.photo = Image.fromarray(self.photo)
        self.photo = ImageTk.PhotoImage(self.photo)

        self.image_label = tk.Label(self.image_frame, image=self.photo)
        self.image_label.pack()

        self.update_button_colors()
        self.update_counter_label()

    def read_results(self):
        self.update_text_view()

    def update_text_view(self):
        current_data = self.data[self.current_index]
        # if current_image in self.classification_results:
        #     class_label = self.classification_results[current_image]
        current_label = current_data[label_seq-1]
        class_name = classes_list[current_label]
        self.selected_image_label.config(text=class_name, fg="red")  # Change text color to green
        # for button in self.class_buttons:
        #     if button["text"] == class_name:
        #         button.configure(bg="green")
        # else:
        #     self.selected_image_label.config(text="")

    def finish_classification(self):
        self.save_results()
        messagebox.showinfo("Info", "Image labeling complete.")
        self.root.destroy()

    def label_image(self, class_name):
        # current_data = self.data[self.current_index]
        self.data[self.current_index][label_seq-1] = classes_list.index(class_name)
        # self.classification_results[current_image] = class_name
        self.selected_image_label.config(text=class_name)
        self.update_button_colors()
        self.next_image()

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.data):
            self.image_label.destroy()
            self.display_image()
            self.back_button.config(state=tk.NORMAL)
        else:
            self.save_results()
            messagebox.showinfo("Info", "Image labeling complete.")
            self.current_index -= 1

            # self.root.destroy()
        self.update_counter_label()

    def previous_image(self):
        self.current_index -= 1
        self.image_label.destroy()
        self.display_image()
        if self.current_index == 0:
            self.back_button.config(state=tk.DISABLED)
        else:
            current_data = self.data[self.current_index]
            current_label = current_data[label_seq - 1]
            class_name = classes_list[current_label]
            # if current_data in self.classification_results:
            #     class_name = self.classification_results[current_data]
            self.selected_image_label.config(text=class_name, fg="red")  # Change text color to green
            # for button in self.class_buttons:
            #     if button["text"] == class_name:
            #         button.configure(bg="green")

        self.update_counter_label()

    def update_button_colors(self):
        # pass
        # for button in self.class_buttons:
        #     button.configure(bg="SystemButtonFace")  # Reset button color

        current_data = self.data[self.current_index]
        current_label = current_data[label_seq-1]
        class_name = classes_list[current_label]
        # if current_data in self.classification_results:
        #     class_name = self.classification_results[current_data]
        self.selected_image_label.config(text=class_name, fg="red")  # Change text color to green
        for button in self.class_buttons:
            if button["text"] == class_name:
                button.configure(bg="green")
            else:
                button.configure(bg="gray")
        # else:
        #     self.selected_image_label.config(text="")

    def show_warning(self):
        result = messagebox.askyesno("Warning", "Are you sure you want to finish labeling?")
        if result == tk.YES:
            self.finish_classification()

    def save_results(self):
        with open(output_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.data)


if __name__ == "__main__":
    root = tk.Tk()
    gui = ImageLabellerGUI(root, input_type)
    root.mainloop()
