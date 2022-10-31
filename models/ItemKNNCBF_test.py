import pandas as pd
import numpy as np

from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Similarity.Compute_Similarity_Python import *
from Utils.evaluation_n_metrics import evaluate
from Utils.prep_sub import write_csv
from Utils.preproc_n_split import preproc_n_split

# Hyperparameter
test_split = 0.2
validation_split = 0.1
shrink = 100000
slice_size = 100

# URM_all_dataframe, ICM_dataframe2, ICM_dataframe = read_data()
data_loader = DataLoaderSplit()
iai = data_loader.get_iai()
train, test, val, n_episode_list, ICM_dataframe = data_loader.get_all_csr_matrices()


evaluator_validation = EvaluatorHoldout(val, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(train, cutoff_list=[10])

# ##############################################	recommender part	 ############################################# #

### recommender without evaluation, fix write_csv

# class ItemKNNCBFRecommender(object):
#
#     def __init__(self, URM, ICM):
#         self.URM = URM
#         self.ICM = ICM
#
#     def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
#         similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink,
#                                                       topK=topK, normalize=normalize,
#                                                       similarity=similarity)
#
#         self.W_sparse = similarity_object.compute_similarity()
#
#     def recommend(self, user_id, at=None, exclude_seen=True):
#         # compute the scores using the dot product
#         user_profile = self.URM[user_id]
#         scores = user_profile.dot(self.W_sparse).toarray().ravel()
#
#         if exclude_seen:
#             scores = self.filter_seen(user_id, scores)
#
#         # rank items
#         ranking = scores.argsort()[::-1]
#
#         return ranking[:at]
#
#     def filter_seen(self, user_id, scores):
#         start_pos = self.URM.indptr[user_id]
#         end_pos = self.URM.indptr[user_id + 1]
#
#         user_profile = self.URM.indices[start_pos:end_pos]
#
#         scores[user_profile] = -np.inf
#
#         return scores
#
#
# recommender = ItemKNNCBFRecommender(train, ICM_dataframe)
# recommender.fit(shrink=100000, topK=50)
#
# for user_id in range(100):
#     print(recommender.recommend(user_id, at=10))
#
# write_csv(iai, train, val, recommender)





from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

x_tick_grid = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
MAP_per_k_grid = []

# for topK in x_tick_grid:
recommender = ItemKNNCBFRecommender(train, ICM_dataframe)
recommender.fit(shrink=100000, topK=50)

result_df, _ = evaluator_test.evaluateRecommender(recommender)

MAP_per_k_grid.append(result_df.loc[10]["MAP"])
print(MAP_per_k_grid[0])



#
# for user_id in range(10):
#     print(recommender.recommend(user_id, at=10))
#
# write_csv(iai, train, val, recommender)


# from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
# recommender = ItemKNNCBFRecommender(train, ICM_dataframe)
# recommender.fit(shrink=1000, topK=50)
# for user_id in range(10):
#     print(recommender.recommend(user_id, at=10))
#
# MAP_per_k_grid = []
# result_df, _ = evaluator_test.evaluateRecommender(recommender)
#
# MAP_per_k_grid.append(result_df.loc[10]["MAP"])
# print(MAP_per_k_grid[0])

# from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
#
#
# MAP_per_shrinkage_grid = []
# recommender = ItemKNNCBFRecommender(train, ICM_dataframe)
# recommender.fit(shrink=10, topK=100)
#
# result_df, _ = evaluator_test.evaluateRecommender(recommender)
#
# MAP_per_shrinkage_grid.append(result_df.loc[10]["MAP"])
# print(result_df.loc[10])

# accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluate(recommender_ItemKNNCBF,
#                                                                                            train,
#                                                                                            test)

# x_tick_grid = [0, 10, 50, 100, 200, 300, 400, 500]
# MAP_per_shrinkage_grid = []
#
# for shrink in x_tick_grid:
#     recommender = ItemKNNCBFRecommender(train, ICM_dataframe)
#     recommender.fit(shrink=shrink, topK=100)
#
#     result_df, _ = evaluator_test.evaluateRecommender(recommender)
#
#     MAP_per_shrinkage_grid.append(result_df.loc[10]["MAP"])
#
# for i in range(0, len(MAP_per_shrinkage_grid)):
#     print(MAP_per_shrinkage_grid[i])

########################################################################################################################

# accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluate(recommender_ItemKNNCBF,
#                                                                                            urm_train,
#                                                                                            urm_test)

# write_csv(URM_all_dataframe, urm_train, urm_validation, recommender_ItemKNNCBF)
