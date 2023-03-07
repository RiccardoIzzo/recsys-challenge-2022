from Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
import implicit.evaluation

URM_train = sps.load_npz('../data/dataset_split/URM_train_new.npz')
URM_train_imp = sps.load_npz('../data/dataset_split/URM_train_imp_new.npz')
URM_test = sps.load_npz('../data/dataset_split/URM_test_new.npz')
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

from skopt import gp_minimize
import implicit.evaluation


# factors 14, reg 0.0005043845626844643, epochs 44
def objective(params):
    # Extract the hyperparameters from the params list
    factors, regularization, iterations = params

    # Initialize the ALS model with the current hyperparameters
    model = als = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                       iterations=iterations)

    # Fit the model to the training data
    model.fit(URM_train)

    # Compute the MAP@k metric on the test data
    map_at_k = implicit.evaluation.mean_average_precision_at_k(model, URM_train, URM_test,
                                                               K=10, show_progress=True,
                                                               num_threads=2)

    # Return the negative of the MAP@k, since skopt minimizes the function
    return -map_at_k


# Define the search space for the hyperparameters
space = [(10, 200), (10 ** -4, 10 ** -2), (40, 100)]

# Use skopt to find the set of hyperparameters that maximizes the MAP@k
best_params = gp_minimize(objective, space, n_calls=100)

print(best_params)

import implicit
import numpy as np


class ALSWrapper:
    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.USER_factors = None  # n_users x n_factors
        self.ITEM_factors = None  # n_items x n_factors
        self.model = None
        self.n_users, self.n_items = URM_train.shape

    def fit(self, factors, regularization, iterations):
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization,
                                                          iterations=iterations)
        self.model.fit(self.URM_train)
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array],
                                                      self.ITEM_factors[items_to_compute, :].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        return item_scores
