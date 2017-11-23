import os

import numpy as np


class LoadDataset:
    def __init__(self):
        pass

    def loading(self):
        url = os.path.dirname(__file__) + "\\dataset\\"

        import pandas as pd
        df = pd.read_csv(url + "aps.csv", sep=';', header=None, dtype=float)
        dataset = df.values
        return dataset

    def normalize(self, data):
        max_range_list = []
        min_range_list = []
        if type(data[0]) == np.ndarray:
            normalized_data = np.empty([len(data), len(data[0])], dtype=float)
            for i in range(len(normalized_data)):
                max_range_list.append(max(data[i]))
                min_range_list.append(min(data[i]))
            max_range = max(max_range_list)
            min_range = min(min_range_list)
            for i in range(len(normalized_data)):
                for j in range(len(normalized_data[0])):
                    normalized_data[i][j] = data[i][j] / float(max_range - min_range)
        else:
            normalized_data = np.empty([len(data), 1], dtype=float)
            max_range = max(data)
            min_range = min(data)
            for i in range(len(normalized_data)):
                normalized_data[i] = data[i] / float(max_range - min_range)
        print("Min: " + str(min_range) + " Max: " + str(max_range))
        return normalized_data