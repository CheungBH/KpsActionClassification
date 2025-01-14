import random
import cv2
import numpy as np


class Augmentor:
    def __init__(self):
        self.shift_range = 0.2
        self.scale_range = 0.1
        self.flip_factor = 0.5
        self.augmented_data = []
        self.augmented_label = []
        self.color_dict = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
        self.color_idx = [4, 7, 10, 13, 16]
        self.augment_factor = []
        self.connection = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (2, 4), (3, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]
        self.classes_name = ["backhand", "forehand", "overhead", "waiting"]
    # def choose_color(self, coord_i):
        # c_idx = 0
        # while True:
        #     if coord_i <= self.color_idx[c_idx]:
        #         return self.color_dict[c_idx]
        #     else:
        #         c_idx += 1
    def choose_color(self, coord_i):
        if coord_i <= 4:
            return self.color_dict[0]
        elif coord_i in [5, 7, 9]:
            return self.color_dict[1]
        elif coord_i in [6, 8, 10]:
            return self.color_dict[2]
        elif coord_i in [11, 13, 15]:
            return self.color_dict[3]
        elif coord_i in [12, 14, 16]:
            return self.color_dict[4]

    def augment(self, data, label):
        for d, l in zip(data, label):
            x_shift = random.uniform(-self.shift_range, self.shift_range)
            y_shift = random.uniform(-self.shift_range, self.shift_range)
            x_scale = random.uniform(-self.scale_range, self.scale_range)
            y_scale = random.uniform(-self.scale_range, self.scale_range)
            flip_x = random.uniform(0, 1)
            self.augmented_data.append(self._augment(d, x_shift, y_shift, x_scale, y_scale, flip_x))
            self.augmented_label.append(l)
            self.augment_factor.append([x_shift, y_shift, x_scale, y_scale, flip_x])

    def _augment(self, data, x_shift, y_shift, x_scale, y_scale, flip_x):
        augmented_data = []
        for i, d in enumerate(data):
            if i % 2 == 0:
                if flip_x > self.flip_factor:
                    d = 1 - d
                augmented_data.append(d - ((d - 0.5) * x_shift) + x_scale)
            else:
                augmented_data.append(d - ((d - 0.5) * y_shift) + y_scale)
            if i > 34:
                break
        return augmented_data

    def append_dataset(self, data, label):
        return data + self.augmented_data, label + self.augmented_label

    def visualize(self, data=None, label=None):
        if data is None or label is None:
            data, label = self.augmented_data, self.augmented_label
        for single_coord, label in zip(data, label):
            if int(label) != 0:
                continue
            image = np.zeros((1000, 1000, 3), dtype=np.uint8)
            float_single_coord = [x * 1000 for x in single_coord]
            cv2.putText(image, self.classes_name[int(label)], (100, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=3, color=(255, 255, 255), thickness=3)

            # print(label)
            for i in range(17):
                x = int(float_single_coord[i*2])
                y = int(float_single_coord[i*2+1])
                cv2.circle(image, (x, y), 5, self.choose_color(i), -1)

            for i, j in self.connection:
                x1, y1 = int(float_single_coord[i * 2]), int(float_single_coord[i * 2 + 1])
                x2, y2 = int(float_single_coord[j * 2]), int(float_single_coord[j * 2 + 1])
                cv2.line(image, (x1, y1), (x2, y2), self.choose_color(i), 2)

            cv2.imshow("coord", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import csv
    file_path = "data/general_ML_bf/train.csv"
    data_feature = []
    data_target = []
    feature_num = 34

    csv_file = csv.reader(open(file_path))
    for content in csv_file:
        content = list(map(float, content[:-1]))
        if len(content) != 0:
            data_feature.append(content[0:feature_num])
            data_target.append(content[feature_num])

    augmentor = Augmentor()
    augmentor.augment(data_feature, data_target)
    # data_feature, data_target = augmentor.append_dataset(data_feature, data_target)
    augmentor.visualize(data_feature, data_target)
    augmentor.visualize()


