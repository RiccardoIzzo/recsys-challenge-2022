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

dataReader = DataLoaderSplit(urm='interactionScoredOld.csv')
URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender

recommender_ItemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_type)
recommender_ItemKNNCFCBF.fit()
result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCFCBF)
print(result_df.loc[10])

best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

ItemKNNCF = ItemKNNCFRecommender(URM_train=URM_train)
ItemKNNCF.fit(**best_hyperparams_ItemKNNCF)
result_dict, _ = evaluator_validation.evaluateRecommender(ItemKNNCF)

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

P3alpha = P3alphaRecommender(URM_train)
P3alpha.fit()

alpha = 0.7
new_similarity = (1 - alpha) * ItemKNNCF.W_sparse + alpha * P3alpha.W_sparse

from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
recommender_object.fit(new_similarity)
result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
print(result_df.loc[10])

from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender

pureSVD = PureSVDRecommender(URM_train)
pureSVD.fit()

from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps


class ScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


#
scoreshybridrecommender = ScoresHybridRecommender(URM_train, ItemKNNCF, pureSVD)
scoreshybridrecommender.fit(alpha=0.5)
#
# # hyperparameterSearch = SearchBayesianSkopt(recommender_class,
# #                                            evaluator_validation=evaluator_validation,
# #                                            evaluator_test=evaluator_test)
#


from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

shrink = 5000000
pureSVD = EASE_R_Recommender(URM_train)
pureSVD.fit()

from Recommenders.NonPersonalizedRecommender import TopPop

TopPop = TopPop(URM_train)
TopPop.fit()

scoreshybridrecommender = ScoresHybridRecommender(URM_train, TopPop, pureSVD)
scoreshybridrecommender.fit(alpha=0.5)

result_df, _ = evaluator_validation.evaluateRecommender(scoreshybridrecommender)
print(result_df.loc[10])


# recommender.save_model('../trained_models', file_name='IALS_recommender_model')

from numpy import linalg as LA

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

    recommender_object = DifferentLossScoresHybridRecommender(URM_train, ItemKNNCF_recommender, P3alpha_recommender)

    for norm in [1, 2, np.inf, -np.inf]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recommender_object.fit(norm, alpha=alpha)

            result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
            print("Norm: {}, Alpha: {}, Result: {}".format(norm, alpha, result_df.loc[10]["MAP"]))
