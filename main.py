import numpy as np
import pandas as pd
# Import PySwarms
from numpy import random
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

import fitness_function as ff
from DragonFlyOptimizer import DragonFlyOptimizer
from InformationGain import FeatureSelection


class Project(FeatureSelection):

    def __init__(self,csv = "australian.csv", num_feature_select = 10):
        super().__init__(csv, num_feature_select)

        # self.csv_data = pd.read_csv("australian.csv")

        self.mutual_info_calculator()
        self.X = self.csv_data.iloc[:, self.top_n_features]

        # self.X = self.csv_data.iloc[:, 0:14]
        self.Y = self.csv_data.iloc[:, 14:15].values.reshape(-1, )
        self.best_ind = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
        self.classifier = GaussianNB()
        # self.classifier = svm.SVC()

    def f_per_particle(self, m, alpha):
        """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
        fit_obj = ff.FitenessFunction()

        total_features = 14
        # Get the subset of the features from the binary mask
        if np.count_nonzero(m) == 0:
            X_subset = self.X
        else:
            feature_idx = np.where(np.asarray(m) == 1)[0]
            X_subset = self.X.iloc[:, feature_idx]

        # print("particle : ", m)
        P = fit_obj.calculate_fitness(self.classifier, X_subset, self.Y)

        # Perform classification and store performance in P
        # classifier.fit(X_subset, self.Y)
        # P = (classifier.predict(X_subset) == self.Y).mean()
        # Compute for the objective function
        j = (alpha * (1.0 - P)
             + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

        # alpha = random.random()
        # beta = 1 - alpha

        # j = alpha * P + beta * (X_subset.shape[1] / total_features)

        return j

    def f(self, x, alpha=0.88):
        """Higher-level method to do classification in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        j = [self.f_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)

    def Optimize(self):
        options = {'c1': 0.3, 'c2': 0.6, 'w': 0.95, 'k': 5, 'p': 2}

        # Call instance of PSO
        dimensions = self.num_feature_select  # dimensions should be the number of features
        # optimizer.reset()
        # optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
        optimizer = DragonFlyOptimizer(n_particles=40, dimensions=dimensions)

        # Perform optimization
        cost, pos = optimizer.optimize(self.f, print_step=5, iters=100, verbose=2)
        self.best_ind = pos
        # print("Cost :: ", cost)
        # print("POS :: ", pos)

    def train(self):
        fit_obj = ff.FitenessFunction(10)
        feature_idx = np.where(np.asarray(self.best_ind) == 1)[0]
        # print(feature_idx, self.best_ind)
        # print(self.X.iloc[:,feature_idx].shape)
        fitness = fit_obj.calculate_fitness(self.classifier, self.X.iloc[:, feature_idx], self.Y)
        print("The accuracy using feature set {} is {}%".format(feature_idx, fitness * 100))


s = Project(num_feature_select=10)
s.Optimize()
s.train()
