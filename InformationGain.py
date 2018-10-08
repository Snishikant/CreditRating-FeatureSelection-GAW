
from __future__ import print_function
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import pandas as pd
from pprint import pprint
import operator



class FeatureSelection:

    def __init__(self, csv, num_feature_select):

        # self.cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'y' ]
        self.cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'y']
        self.num_cols = len(self.cols)
        self.information_gain = {}     # Information gain for all features numbered 0 - (n - 1)
        self.num_feature_select = num_feature_select    # Number of top features to select
        self.top_n_features = []       # Top n features

        self.discrete_features = [0, 3, 4, 5, 7, 8, 10, 11]    # Features having discrete Values
        self.csv_data = pd.read_csv(csv, names = self.cols)
        self.X = self.csv_data.iloc[:, 0:14];
        self.Y = self.csv_data.iloc[:, 14: 15].values.reshape(-1,)



        # self.gain = {}
        # print(self.Y.values.reshape(-1,))



    def mutual_info_calculator(self):
        information_gain = []
        information_gain.append(mutual_info_regression(self.X, self.Y,  discrete_features = self.discrete_features))

        info_gain = {}

        for i in range(self.X.shape[1]):
            info_gain[str(i)] = information_gain[0][i]

        info_gain = sorted(info_gain.items(), key=operator.itemgetter(1), reverse = True)

        for i in range(self.X.shape[1]):
            if i < self.num_feature_select:
                self.top_n_features.append(int(info_gain[i][0]))
            self.information_gain[info_gain[i][0]] = info_gain[i][1]


# p = FeatureSelection("australian.csv", 10)
# p.mutual_info_calculator()
# print(p.information_gain)
# print(p.top_n_features)
