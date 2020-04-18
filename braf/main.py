import os, sys, shutil
import pandas as pd
import numpy as np
from random import seed
from random import randrange
from .algorithms import NearestNeighbors
from .algorithms.RandomForest import accuracy_metric, cross_validation_split, decision_tree

class BiasedRandomForestClassifier:
    """
    Bias Random Forest Classifier extends the logic underlying Random Forest Classifier to address issue of minority labels
    inproportionately representing the random forest classification process. By moving the oversampling from the data level
    to the algorithm level, the Biased Random Forest Classifier aims to create a classification ensemble process agnostic
    of the data.

    # Step 1
    The nearest neighbor algorithm is employed to identify the difficult/critical areas in the data set, which are the
    minority instances and their k-nearest majority neighbors.

    # Step 2
    Standard Random Forest is generated from all records in the data set.

    # Step 3
    The standard random forest is fed with more random trees generated based on the difficult areas, to address
    1) more diverse ensemble/forest
    2) biased towards the minority class

    """

    def __init__(self, df, columns, s, p, K, *args, **kwargs):
        super(BiasedRandomForestClassifier, self).__init__()
        self.df = df
        self.col_names = columns
        self.s = s  # random forest size
        self.p = p  # ratio of critical areas
        self.K = K  # number of nearest neighbors

    def maj_min_split(self, label=0):
        df = pd.DataFrame(np.row_stack(self.df), columns = self.col_names)

        # Identify Major Labels and Minor Labels
        train_major = df[df['Outcome'] == label]
        train_minor = df[df['Outcome'] != label]

        return train_minor.values.tolist()

    def make_crit_df(self):
        seed(1)
        minority_set = self.maj_min_split(label=0)

        df = pd.DataFrame(np.row_stack(self.df), columns=self.col_names)
        df = df.values.tolist()

        crit_df = []
        for test_row in minority_set:
            crit_df.append(test_row)
            Tnn = NearestNeighbors.get_neighbors(df,
                                                 test_row,
                                                 num_neighbors=self.K)

            for tn in Tnn:
                if tn not in crit_df:
                    crit_df.append(tn)

        return crit_df

    def combine_forests(self):
        """
        Combines results from standard random forest and the trees sampled from nearest neighbor
        # Set K = 10, p = 0.5, s = 100
        s: size
        K: number of nearest neighbors
        p: proportion of samples from critical areas
        :return:
        """

        def build_forest(df, algorithm, n_folds, max_depth, min_size):
            folds = cross_validation_split(df, n_folds)
            scores = list()
            values = dict()
            values['actual'] = []
            values['predicted'] = []
            for fold in folds:
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()
                for row in fold:
                    row_copy = list(row)
                    test_set.append(row_copy)
                    row_copy[-1] = None
                predicted = algorithm(train_set, test_set, max_depth, min_size)
                actual = [row[-1] for row in fold]
                accuracy = accuracy_metric(actual, predicted)
                scores.append(accuracy)
                values['actual'].append(actual)
                values['predicted'].append(predicted)

            return scores, values

        df = self.df
        df = pd.DataFrame(np.row_stack(df), columns=self.col_names)

        seed(1)
        sub_df = df.sample(int(len(df) * (1 - self.p)))
        sub_df = sub_df.values.tolist()

        col_names = df.columns.tolist()

        seed(1)
        crit_df = self.make_crit_df()
        sub_crit = pd.DataFrame(crit_df, columns=col_names).sample(int(len(df) * self.p))
        sub_crit.values.tolist()

        rf1_scores, rf1_values = build_forest(df = sub_df,
                                              algorithm = decision_tree,
                                              n_folds = self.K,
                                              max_depth = 10,
                                              min_size = self.s)

        rf2_scores, rf2_values = build_forest(df = crit_df,
                                       algorithm = decision_tree,
                                       n_folds = self.K,
                                       max_depth = 10,
                                       min_size = self.s)

        braf_values = {**rf1_values, **rf2_values}

        values = {}
        values['values'] = {}
        values['values']['full_data'] = rf1_values
        values['values']['crit_data'] = rf2_values
        values['values']['braf'] = braf_values

        #return scores, values
        return values