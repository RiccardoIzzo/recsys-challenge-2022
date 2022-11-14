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
from Utils.prep_sub import write_submission

dataReader = DataLoaderSplit(urm='LastURM.csv')
URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

### INSERT YOUR RECOMMENDERS WITH THE BEST HYPERPARAMS HERE ###


from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

best_hyperparams_ItemKNNCF = {'topK': 135, 'shrink': 257, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}
ItemKNNCF = ItemKNNCFRecommender(URM_train=URM_train)
ItemKNNCF.fit(**best_hyperparams_ItemKNNCF)
result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF)
print("{} FINAL MAP: {}".format(ItemKNNCF.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
print("Recommender 1 is ready!")

# from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
# best_hyperparams_EASE_R = {'topK': None, 'normalize_matrix': False, 'l2_norm': 6.684323468481221}
# EASE_R = EASE_R_Recommender(URM_train)
# EASE_R.fit(**best_hyperparams_EASE_R)
# result_df, _ = evaluator_validation.evaluateRecommender(EASE_R)
# print("{} FINAL MAP: {}".format(EASE_R.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
# print("Recommender 2 is ready!")


from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

best_hyperparams_SLIM_BPR_Cython = {'topK': 999, 'epochs': 320, 'symmetric': True, 'sgd_mode': 'sgd',
                                    'lambda_i': 0.00014089714882493655, 'lambda_j': 4.0389558749691306e-05,
                                    'learning_rate': 0.0081640727301237461}
SLIM_BPR_Cython = SLIM_BPR_Cython(URM_train)
SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)
result_df, _ = evaluator_validation.evaluateRecommender(SLIM_BPR_Cython)
print("{} FINAL MAP: {}".format(SLIM_BPR_Cython.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
print("Recommender 2 is ready!")


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


recommender_object = DifferentLossScoresHybridRecommender(URM_train, ItemKNNCF, SLIM_BPR_Cython)

for norm in [1, 2, np.inf, -np.inf]:
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        recommender_object.fit(norm, alpha=alpha)

        result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
        print("Norm: {}, Alpha: {}, Result: {}".format(norm, alpha, result_df.loc[10]["MAP"]))


# recommender_object.fit(norm=2, alpha=0.9)
# result_dict, _ = evaluator_validation.evaluateRecommender(recommender_object)
# print(result_dict.MAP)


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


# scoreshybridrecommender = ScoresHybridRecommender(URM_train, ItemKNNCF, pureSVD)
# scoreshybridrecommender.fit(alpha=0.5)


# write_submission(recommender=recommender_object)
