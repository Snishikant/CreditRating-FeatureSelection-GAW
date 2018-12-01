from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class FitenessFunction:

    def __init__(self,n_splits = 5,*args,**kwargs):
        """
            Parameters
            -----------
            n_splits :int,
                Number of splits for cv

            verbose: 0 or 1
        """
        self.n_splits = n_splits


    def calculate_fitness(self,model,x,y):

        # x = x.values.

        cv_set = np.repeat(-1.,x.shape[0])
        skf = StratifiedKFold(n_splits = self.n_splits)

        for train_index,test_index in skf.split(x,y):
            # print(train_index, test_index)


            x_train,x_test = x.iloc[train_index],x.iloc[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train, y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y

        return accuracy_score(y,cv_set)

    def give_predicted(self,model,x,y):
        cv_set = np.repeat(-1.,x.shape[0])
        skf = StratifiedKFold(n_splits = self.n_splits)

        for train_index,test_index in skf.split(x,y):
            # print(train_index, test_index)


            x_train,x_test = x.iloc[train_index],x.iloc[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train, y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y
        return cv_set
