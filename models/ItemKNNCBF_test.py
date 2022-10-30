import pandas as pd
from Utils.DataLoaderSplit import DataLoaderSplit
from Utils.evaluation_n_metrics import evaluate
from Utils.prep_sub import write_csv
from Utils.preproc_n_split import preproc_n_split

# Hyperparameter
test_split = 0.2
validation_split = 0.1
shrink = 350000000
slice_size = 100

# URM_all_dataframe, ICM_dataframe2, ICM_dataframe = read_data()
data_loader = DataLoaderSplit()
train, test, val, n_episode_list, ICM_dataframe = data_loader.get_csr_matrices()
"""
URM_all_dataframe, num_users, num_items, urm_train, urm_validation, urm_test = preproc_n_split(URM_all_dataframe, test_split=test_split,
                                                                                 val_split=validation_split)

mapped_id, original_id = pd.factorize(URM_all_dataframe["user_id"].unique())

print("Unique UserID in the URM are {}".format(len(original_id)))

user_original_ID_to_index = pd.Series(mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(URM_all_dataframe["item_id"].unique())

print("Unique ItemID in the URM are {}".format(len(original_id)))

all_item_indices = pd.concat([URM_all_dataframe["item_id"], ICM_dataframe["item_id"]], ignore_index=True)
mapped_id, original_id = pd.factorize(all_item_indices.unique())

print("Unique ItemID in the URM and ICM are {}".format(len(original_id)))

item_original_ID_to_index = pd.Series(mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(ICM_dataframe["feature_id"].unique())
feature_original_ID_to_index = pd.Series(mapped_id, index=original_id)

print("Unique FeatureID in the URM are {}".format(len(feature_original_ID_to_index)))

URM_all_dataframe["user_id"] = URM_all_dataframe["user_id"].map(user_original_ID_to_index)
URM_all_dataframe["item_id"] = URM_all_dataframe["item_id"].map(item_original_ID_to_index)
"""




# import numpy as np
# import scipy.sparse as sp
# from similarity_functions.cosine_similarity_SLOW import cosine_similarity
# from typing import Optional




import scipy.sparse as sps

"""

ICM_genres = genre_list["feature_id"]
stacked_URM = sps.vstack([urm_train, ICM_genres.T])
stacked_URM = sps.csr_matrix(stacked_URM)
stacked_ICM = sps.csr_matrix(stacked_URM.T)
"""

# n_users = len(user_original_ID_to_index)
# n_items = len(item_original_ID_to_index)
# n_features = len(feature_original_ID_to_index)
# ICM_all = sps.csr_matrix(
#     (np.ones(len(ICM_dataframe["item_id"].values)), (ICM_dataframe["item_id"].values, ICM_dataframe["feature_id"].values)),
#     shape=(n_items, n_features))
#
# ICM_all.data = np.ones_like(ICM_all.data)

# ##############################################	recommender part	 ############################################# #

from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

recommender_ItemKNNCBF = ItemKNNCBFRecommender(URM_all_dataframe, ICM_dataframe)

recommender_ItemKNNCBF.fit()

########################################################################################################################

# accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluate(recommender_ItemKNNCBF,
#                                                                                            urm_train,
#                                                                                            urm_test)

#write_csv(URM_all_dataframe, urm_train, urm_validation, recommender_ItemKNNCBF)
