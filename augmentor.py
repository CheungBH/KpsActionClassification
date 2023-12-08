import random


class Augmentor:
    def __init__(self):
        self.shift_range = 0.2
        self.scale_range = 0.1
        self.augmented_data = []
        self.augmented_label = []

    def augment(self, data, label):
        for d, l in zip(data, label):
            x_shift = random.uniform(-self.shift_range, self.shift_range)
            y_shift = random.uniform(-self.shift_range, self.shift_range)
            x_scale = random.uniform(-self.shift_range, self.shift_range)
            y_scale = random.uniform(-self.shift_range, self.shift_range)
            self.augmented_data.append(self._augment(d, x_shift, y_shift, x_scale, y_scale))
            self.augmented_label.append(l)

    def _augment(self, data, x_shift, y_shift, x_scale, y_scale):
        augmented_data = []
        for i, d in enumerate(data):
            if i % 2 == 0:
                augmented_data.append(d - ((d - 0.5) * x_shift) + x_scale)
            else:
                augmented_data.append(d - ((d - 0.5) * y_shift) + y_scale)
        return augmented_data

    def append_dataset(self, data, label):
        return data + self.augmented_data, label + self.augmented_label

    def visualize(self):
        pass