import numpy as np
import pandas as pd


def information_gain(x, y):

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
    print(x.T.shape)
    for feature in x.T:
    	# print(feature)
        information_gain_scores.append(ig(feature, y))
    return information_gain_scores, []

aus = pd.read_csv("australian.csv")
dis = [0,3,4,5,7,8,10,11]
X = aus.iloc[:,0:14]
Y = aus.iloc[:,14:15]
# print(X.shape, Y.shape)

print(Y, "\n"*5)
print(Y.values[:,0])

# from numpy import genfromtxt
# my_data = genfromtxt('australian.csv', delimiter=',')
# X = np.empty
# for

print(information_gain(X.values,Y.values[:,0]))
