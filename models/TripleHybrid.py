from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Hybrid
from Recommenders.NonPersonalizedRecommender import *
from run_all_algorithms import _get_instance
from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
from Utils.prep_sub import write_submission
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from numpy import linalg as LA
import numpy as np
from Utils.prep_sub import write_submission
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Integer, Categorical, Real


def main():
    dataReader = DataLoaderSplit(urm='newURM.csv')
    URM_all = dataReader.get_implicit_single_mixed_csr()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
    # evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

    best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
                                'normalize_similarity': True, 'implicit': True}
    RP3beta = RP3betaRecommender(URM_train=URM_train)
    RP3beta.fit(**best_hyperparams_RP3beta)
    result_df, _ = evaluator_validation.evaluateRecommender(RP3beta)
    print("{} FINAL MAP: {}".format(RP3beta.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

    best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
    EASE_R = EASE_R_Recommender(URM_train=URM_train)
    EASE_R.fit(**best_hyperparams_EASE_R)
    result_df, _ = evaluator_validation.evaluateRecommender(EASE_R)
    print("{} FINAL MAP: {}".format(EASE_R.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

    best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
                                                  'alpha': 0.0009354394779247897}
    SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=URM_train)
    SLIMElasticNet.fit(**best_hyperparams_SLIMElasticNetRecommender)
    result_df, _ = evaluator_validation.evaluateRecommender(SLIMElasticNet)
    print("{} FINAL MAP: {}".format(SLIMElasticNet.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    class ScoresHybridRecommender_3(BaseRecommender):
        RECOMMENDER_NAME = "ScoresHybridRecommender_3"

        def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
            super(ScoresHybridRecommender_3, self).__init__(URM_train)

            self.alpha = None
            self.beta = None
            self.norm = None

            self.URM_train = sps.csr_matrix(URM_train)
            self.recommender_1 = recommender_1
            self.recommender_2 = recommender_2
            self.recommender_3 = recommender_3

        def fit(self, norm, alpha=0.5, beta=0.5):
            self.norm = norm
            self.alpha = alpha
            self.beta = beta

        def _compute_item_score(self, user_id_array, items_to_compute):
            # In a simple extension this could be a loop over a list of pretrained recommender objects
            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
            item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
            norm_item_weights_3 = LA.norm(item_weights_3, self.norm)

            item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha * self.beta + (item_weights_2 / norm_item_weights_2) * self.alpha * (1 - self.beta) + (item_weights_3 / norm_item_weights_3) * (1 - self.alpha)

            # item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha + (
            #         item_weights_2 / norm_item_weights_2) * self.beta + (item_weights_3 / norm_item_weights_3) * (
            #                        1 - self.alpha - self.beta)

            return item_weights

        def save_model(self, folder_path, file_name=None):
            return

        def load_model(self, folder_path, file_name=None):
            return

    hyperparameters_range_dictionary = {
        "norm": Categorical([1, 2, np.inf, -np.inf]),
        "alpha": Real(low=0.0, high=1.0, prior='uniform'),
        "beta": Real(low=0.0, high=1.0, prior='uniform'),
    }

    recommender_class = ScoresHybridRecommender_3

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=None)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, SLIMElasticNet, EASE_R, RP3beta],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )
    # recommender_input_args_last_test = SearchInputRecommenderArgs(
    #     CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, SLIMElasticNet, EASE_R, RP3beta],
    #     CONSTRUCTOR_KEYWORD_ARGS={},
    #     FIT_POSITIONAL_ARGS=[],
    #     FIT_KEYWORD_ARGS={}
    # )

    output_folder_path = "../trained_models/"
    n_cases = 250
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    hyperparameterSearch.search(recommender_input_args,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="best",
                                resume_from_saved=False,
                                output_folder_path=output_folder_path,  # Where to save the results
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )


if __name__ == "__main__":
    main()
