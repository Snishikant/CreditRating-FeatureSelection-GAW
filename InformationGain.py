from __future__ import print_function
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import pandas as pd
from pprint import pprint
import operator

import numpy as np


class FeatureSelection:

    def __init__(self, csv, num_feature_select):

        # self.cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'y' ]
        # self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'y']
        # self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 'y']
        self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 'y']
        self.num_cols = len(self.cols)
        self.information_gain = {}  # Information gain for all features numbered 0 - (n - 1)
        self.num_feature_select = num_feature_select  # Number of top features to select
        self.top_n_features = []  # Top n features

        self.discrete_features = [0, 3, 4, 5, 7, 8, 10, 11]    # Features having discrete Values
        # self.discrete_features = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]  # Features having discrete Values
        self.csv_data = pd.read_csv(csv, names=self.cols)
        # self.X = self.csv_data.iloc[:, 0:20];
        # self.Y = self.csv_data.iloc[:, 20: 21].values.reshape(-1,)
        self.X = self.csv_data.iloc[:, 0:14]
        self.Y = self.csv_data.iloc[:, 14:15]

        # self.gain = {}
        # print(self.Y.values.reshape(-1,))

    def exp_IG(self):
        x = self.X.values
        y = self.Y.values[:, 0]

        def _entropy(values):
            counts = np.bincount(values)
            probs = counts[np.nonzero(counts)] / float(len(values))
            # print(1 - probs, probs)
            return np.sum(probs * np.exp(1 - probs))

        def ig(feature, y):
            feature_set_indices = np.nonzero(feature)
            feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices[0]]
            entropy_x_set = _entropy(y[feature_set_indices])
            entropy_x_not_set = _entropy(y[feature_not_set_indices])

            return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                     + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

        feature_size = x.shape[0]
        feature_range = range(0, feature_size)
        # print(feature_size)
        # print(feature_range)
        entropy_before = _entropy(y)
        # print(entropy_before)
        information_gain_scores = []
        # print(x.T.shape)
        for feature in x.T:
            # print(feature)
            information_gain_scores.append(ig(feature, y))
        # print(information_gain_scores)
        info_gain = {}

        for i in range(self.X.shape[1]):
            info_gain[str(i)] = information_gain_scores[i]

        info_gain = sorted(info_gain.items(), key=operator.itemgetter(1), reverse=True)

        for i in range(self.X.shape[1]):
            if i < self.num_feature_select:
                self.top_n_features.append(int(info_gain[i][0]))
            self.information_gain[info_gain[i][0]] = info_gain[i][1]

        # return information_gain_scores, []

    def mutual_info_calculator(self):
        information_gain = []
        information_gain.append(mutual_info_regression(self.X, self.Y, discrete_features=self.discrete_features))

        info_gain = {}

        for i in range(self.X.shape[1]):
            info_gain[str(i)] = information_gain[0][i]

        info_gain = sorted(info_gain.items(), key=operator.itemgetter(1), reverse=True)

        for i in range(self.X.shape[1]):
            if i < self.num_feature_select:
                self.top_n_features.append(int(info_gain[i][0]))
            self.information_gain[info_gain[i][0]] = info_gain[i][1]


# p = FeatureSelection("GermanData.csv", 10)
p = FeatureSelection("australian.csv", 10)

p.exp_IG()
print(p.information_gain)
print(p.top_n_features)
