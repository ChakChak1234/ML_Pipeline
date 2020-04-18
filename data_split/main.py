from random import randrange

class DataSplitProcedures:
    def __init__(self, df, n_folds, *args, **kwargs):
        super(DataSplitProcedures, self).__init__()
        self.df = df
        self.n_folds = n_folds

    def train_test_split(self):
        """
        This function splits the data into training and testing sets
        :return:
        """
        df = self.df
        # shuffle dataset
        shuffle_df = df.sample(frac=1)

        # define size of training set
        train_size = int(0.85 * len(df))

        # split dataset
        train_set = shuffle_df[:train_size].values.tolist()
        test_set = shuffle_df[train_size:].values.tolist()

        return train_set, test_set

    def k_fold_crossvalidation(self):
        """
        This function performs k-fold cross validation split

        train set = k-1 fold of the data
        test set = 1 fold of the data
        :return:
        """
        # shuffle dataset
        df = self.df.sample(frac=1)

        # define number of folds
        n_folds = self.n_folds

        # split dataset using 10-fold cross-validation
        dataset_split = list()
        dataset_copy = df.values.tolist()
        fold_size = int(len(df) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)

        kfold_train_set = dataset_split[:-1]
        kfold_test_set = dataset_split[-1]

        return kfold_train_set, kfold_test_set

    def run(self):
        """
        Run all
        :return:
        """
        train_test, kfold_crossval = self.train_test_split(), self.k_fold_crossvalidation()

        return train_test, kfold_crossval