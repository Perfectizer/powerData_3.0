import numpy as np


class ProcessData:
    def __init__(self):
        pass

    # normalization data using (x-min)/(max-min)
    def normalize_data_min_min_max(self, data):
        min_value = min(data)
        max_value = max(data)
        return [(float(i) - min_value) / float(max_value - min_value) for i in data]

    # normalization data using (x-mean)/(max-min)
    def normalize_data_mean_min_max(self, data):
        mean_value = np.mean(data)
        min_value = min(data)
        max_value = max(data)
        return [(float(i) - mean_value) / (max_value - min_value) for i in data]

    def normalize_data_z_score(self, data):
        mean_value = np.mean(data)
        s2 = sum([(i - mean_value) * (i - mean_value) for i in data]) / len(data) ** 0.5
        return [(i - mean_value) / s2 for i in data]

    def cluster_data(self, data):
        pass
