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
    # FOR GRID SEARCH
    dataReader = DataLoaderSplit(urm='LastURM.csv')
    URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

    # FOR BAYESIAN OPTIMIZATION WITH SKOPT
    # dataReader = DataLoaderSplit(urm='LastURM.csv')
    # URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
    # URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
    # evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    # evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    ### INSERT YOUR RECOMMENDERS WITH THE BEST HYPERPARAMS HERE ###



    # ItemKNNCFRecommender
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    best_hyperparams_ItemKNNCF = {'topK': 135, 'shrink': 257, 'similarity': 'cosine', 'normalize': True,
                                  'feature_weighting': 'TF-IDF'}
    ItemKNNCF = ItemKNNCFRecommender(URM_train=URM_train)
    ItemKNNCF.fit(**best_hyperparams_ItemKNNCF)
    result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF)
    print("{} FINAL MAP: {}".format(ItemKNNCF.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))



    # MultiThreadSLIM_SLIMElasticNetRecommender
    # from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    # best_hyperparams_SLIMElasticNetRecommender = {'topK': 2476, 'l1_ratio': 5.06543030058637e-06,
    #                                               'alpha': 0.00026049562591808496}
    # SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=URM_train)
    # SLIMElasticNet.fit(**best_hyperparams_SLIMElasticNetRecommender)
    # result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF)
    # print("{} FINAL MAP: {}".format(ItemKNNCF.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))



    # EASE_R_Recommender
    # from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
    #
    # best_hyperparams_EASE_R = {'topK': None, 'normalize_matrix': False, 'l2_norm': 6.684323468481221}
    #
    # EASE_R = EASE_R_Recommender(URM_train=URM_train)
    # EASE_R.fit(**best_hyperparams_EASE_R)
    # result_df, _ = evaluator_validation.evaluateRecommender(EASE_R)
    # print("{} FINAL MAP: {}".format(EASE_R.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))



    # SLIM_BPR_Cython
    # from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    # best_hyperparams_SLIM_BPR_Cython = {'topK': 999, 'epochs': 320, 'symmetric': True, 'sgd_mode': 'sgd',
    #                                     'lambda_i': 0.00014089714882493655, 'lambda_j': 4.0389558749691306e-05,
    #                                     'learning_rate': 0.0081640727301237461}
    # SLIM_BPR_Cython = SLIM_BPR_Cython(URM_train)
    # SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)
    # result_df, _ = evaluator_test.evaluateRecommender(SLIM_BPR_Cython)
    # print("{} FINAL MAP: {}".format(SLIM_BPR_Cython.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))



    # IALS
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log', 'alpha': 38.524701045378585, 'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
    IALS = IALSRecommender(URM_train=URM_train)
    IALS.fit(**best_hyperparams_IALS)
    result_df, _ = evaluator_validation.evaluateRecommender(IALS)
    print("{} FINAL MAP: {}".format(IALS.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))



    # P3alpha
    # from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    # best_hyperparams_P3alpha = {'topK': 633, 'alpha': 0.0, 'normalize_similarity': True}
    # P3alpha = P3alphaRecommender(URM_train=URM_train)
    # P3alpha.fit(**best_hyperparams_P3alpha)
    # result_df, _ = evaluator_validation.evaluateRecommender(P3alpha)
    # print("{} FINAL MAP: {}".format(P3alpha.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))


    #RP3beta
    # from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    # best_hyperparams_RP3beta = {'topK': 1956, 'alpha': 0.0, 'beta': 0.27826585836893003, 'normalize_similarity': True}
    # RP3beta = RP3betaRecommender(URM_train=URM_train)
    # RP3beta.fit(**best_hyperparams_RP3beta)
    # result_df, _ = evaluator_validation.evaluateRecommender(RP3beta)
    # print("{} FINAL MAP: {}".format(RP3beta.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))





    ### HYBRID FUNCTIONS (CHECK ALSO Linear_Combination_Hybrids.py) ###

    ### MODELS WITH RATING PREDICTION VS RANKING LOSS FUNCTION ###
    class DifferentLossScoresHybridRecommender(BaseRecommender):
        """ ScoresHybridRecommender
        Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
        algorithms trained on different loss functions.

        """

        RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

        def __init__(self, URM_train, recommender_1, recommender_2):
            super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

            self.URM_train = sps.csr_matrix(URM_train)
            self.recommender_1 = recommender_1
            self.recommender_2 = recommender_2

        def fit(self, norm, alpha=0.5):

            self.alpha = alpha
            self.norm = norm

        def _compute_item_score(self, user_id_array, items_to_compute):

            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

            item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
                    1 - self.alpha)

            return item_weights



    # GRID SEARCH (norm, alpha)
    recommender_object = DifferentLossScoresHybridRecommender(URM_train, ItemKNNCF, IALS)

    for norm in [1, 2, np.inf, -np.inf]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recommender_object.fit(norm, alpha=alpha)

            result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
            print("Norm: {}, Alpha: {}, Result: {}".format(norm, alpha, result_df.loc[10]["MAP"]))


    # final fit with URM_all + csv
    # recommender_object = DifferentLossScoresHybridRecommender(URM_all, ItemKNNCF, IALS)

    # recommender_object.fit(norm=?, alpha=?)
    # write_submission(recommender=recommender_object)




    # SEARCH WITH SKOPT (norm, alpha)
    # hyperparameters_range_dictionary = {
    #     "norm": Categorical([1, 2, np.inf, -np.inf]),  # Only for DifferentLossScoresHybridRecommender
    #     "alpha": Real(low=0.0, high=1.0, prior='uniform'),
    # }
    #
    # from Utils.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
    # recommender_class = DifferentLossScoresHybridRecommender
    #
    # hyperparameterSearch = SearchBayesianSkopt(recommender_class,
    #                                            evaluator_validation=evaluator_validation,
    #                                            evaluator_test=evaluator_test)
    #
    # recommender_input_args = SearchInputRecommenderArgs(
    #     CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, ItemKNNCF, IALS],
    #     # For a CBF model simply put [URM_train, ICM_train]
    #     CONSTRUCTOR_KEYWORD_ARGS={},
    #     FIT_POSITIONAL_ARGS=[],
    #     FIT_KEYWORD_ARGS={}
    # )
    # recommender_input_args_last_test = SearchInputRecommenderArgs(
    #     CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation, ItemKNNCF, IALS],  # CBF: [URM, ICM], CF: [URM]
    #     CONSTRUCTOR_KEYWORD_ARGS={},
    #     FIT_POSITIONAL_ARGS=[],
    #     FIT_KEYWORD_ARGS={}
    # )
    #
    # output_folder_path = "../trained_models/"
    # n_cases = 30
    # n_random_starts = int(n_cases * 0.3)
    # metric_to_optimize = "MAP"
    # cutoff_to_optimize = 10
    #
    # hyperparameterSearch.search(recommender_input_args,
    #                             recommender_input_args_last_test=recommender_input_args_last_test,
    #                             hyperparameter_search_space=hyperparameters_range_dictionary,
    #                             n_cases=n_cases,
    #                             n_random_starts=n_random_starts,
    #                             save_model="best",
    #                             resume_from_saved=False,
    #                             output_folder_path=output_folder_path,  # Where to save the results
    #                             output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
    #                             metric_to_optimize=metric_to_optimize,
    #                             cutoff_to_optimize=cutoff_to_optimize,
    #                             )

# necessary for multiprocessing (MultiThreadSLIM ...)
if __name__ == "__main__":
    main()





### HYBRIDS WITH THE SAME STRUCTURE ###
# from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
#
# hybrid = ItemKNNCustomSimilarityRecommender(URM_train)
# alpha = 0.3
# new_similarity = (1 - alpha) * ItemKNNCF_recommender.W_sparse + alpha * P3alpha_recommender.W_sparse
# hybrid.fit(new_similarity)
# result_df, _ = evaluator_test.evaluateRecommender(hybrid)
# print(result_df.loc[10])


### HYBRID OF MODELS WITH DIFFERENT STRUCTURE ###
# class ScoresHybridRecommender(BaseRecommender):
#     """ ScoresHybridRecommender
#     Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
#
#     """
#
#     RECOMMENDER_NAME = "ScoresHybridRecommender"
#
#     def __init__(self, URM_train, recommender_1, recommender_2):
#         super(ScoresHybridRecommender, self).__init__(URM_train)
#
#         self.URM_train = sps.csr_matrix(URM_train)
#         self.recommender_1 = recommender_1
#         self.recommender_2 = recommender_2
#
#     def fit(self, alpha=0.5):
#         self.alpha = alpha
#
#     def _compute_item_score(self, user_id_array, items_to_compute):
#         # In a simple extension this could be a loop over a list of pretrained recommender objects
#         item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
#         item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
#
#         item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)
#
#         return item_weights

# scoreshybridrecommender = ScoresHybridRecommender(URM_train, ItemKNNCF, pureSVD)
# scoreshybridrecommender.fit(alpha=?)
# write_submission(recommender=recommender_object)
