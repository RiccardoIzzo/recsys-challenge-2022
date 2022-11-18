from Data_manager.split_functions.split_train_validation_random_holdout import \
	split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Utils.DataLoaderSplit import DataLoaderSplit

import numpy as np


dataReader = DataLoaderSplit(urm='LastURM.csv')
URM, ICM_length, ICM_type = dataReader.get_csr_matrices()

URM, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.8)

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

# ItemKNN CF

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

best_hyperparams_ItemKNNCF = {'topK': 135, 'shrink': 257, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}

ItemKNNCF = ItemKNNCFRecommender(URM_train=URM)
ItemKNNCF.fit(**best_hyperparams_ItemKNNCF)

result_df, _ = evaluator_test.evaluateRecommender(ItemKNNCF)
print("{} FINAL MAP: {}".format(ItemKNNCF.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
print("Recommender 1 is ready!")

# EASE

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

best_hyperparams_EASE_R = {'topK': None, 'normalize_matrix': False, 'l2_norm': 6.684323468481221}

EASE_R = EASE_R_Recommender(URM)
EASE_R.fit(**best_hyperparams_EASE_R)

result_df, _ = evaluator_test.evaluateRecommender(EASE_R)
print("{} FINAL MAP: {}".format(EASE_R.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
print("Recommender 2 is ready!")

# Build Hybrid
from Utils.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender


recommender = DifferentLossScoresHybridRecommender(URM, ItemKNNCF, EASE_R)

map_list = []

for norm in [1, 2, np.inf, -np.inf]:
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        recommender.fit(norm=norm, alpha=alpha)

        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        map_list.append([norm, alpha, result_df.loc[10]["MAP"]])
        print("Norm: ", norm, "Alpha: ", alpha, "Result: ", result_df.loc[10]["MAP"])
