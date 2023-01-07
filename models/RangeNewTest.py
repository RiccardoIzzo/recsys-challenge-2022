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
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Integer, Categorical, Real


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

		item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute)
		item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute)

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


from Utils.DataLoaderSplit import DataLoaderSplit

data_loader = DataLoaderSplit(urm='newURM.csv')
urm_train_scored = sps.load_npz('../LoadModels80/URM_train_new.npz')
urm_train = sps.load_npz('../LoadModels80/URM_train_imp_new.npz')
urm_test = sps.load_npz('../LoadModels80/URM_test_new.npz')
evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], verbose=True)

rec = []
rec_names = []

# ItemKNNCFRecommender
# asymmetric
# from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
# best_hyperparams_ItemKNNCF_asymmetric = {'topK': 50, 'shrink': 77, 'similarity': 'asymmetric', 'normalize': True,
# 										 'asymmetric_alpha': 0.3388995614775971, 'feature_weighting': 'TF-IDF'}
# ItemKNNCF_asymmetric = ItemKNNCFRecommender(URM_train=URM_train)
# ItemKNNCF_asymmetric.fit(**best_hyperparams_ItemKNNCF_asymmetric)
# result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF_asymmetric)
# print("{} FINAL MAP: {}".format(ItemKNNCF_asymmetric.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

# tversky
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

best_hyperparams_ItemKNNCF_tversky = {'topK': 50, 'shrink': 0, 'similarity': 'tversky', 'normalize': True,
									  'tversky_alpha': 0.2677659937917337, 'tversky_beta': 1.8732432514946602}
ItemKNNCF_tversky = ItemKNNCFRecommender(URM_train=urm_train)
ItemKNNCF_tversky.fit(**best_hyperparams_ItemKNNCF_tversky)
rec_names.append(ItemKNNCF_tversky.RECOMMENDER_NAME)
rec.append(ItemKNNCF_tversky)

# SLIMElasticNetRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
											  'alpha': 0.0009354394779247897}
SLIMElasticNet = SLIMElasticNetRecommender(URM_train=urm_train)
SLIMElasticNet.load_model(folder_path="../LoadModels80/",
						  file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_80_new.zip")
rec_names.append(SLIMElasticNet.RECOMMENDER_NAME)
rec.append(SLIMElasticNet)

# EASE_R_Recommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
EASE_R = EASE_R_Recommender(URM_train=urm_train)
EASE_R.load_model(folder_path="../LoadModels80/", file_name=EASE_R.RECOMMENDER_NAME + "_best_80_new")
rec_names.append(EASE_R.RECOMMENDER_NAME)
rec.append(EASE_R)

# IALS
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log', 'alpha': 38.524701045378585,
						 'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
IALS = IALSRecommender(URM_train=urm_train_scored)
IALS.load_model(folder_path="../LoadModels80/", file_name=IALS.RECOMMENDER_NAME + "_best_80_new")
rec_names.append(IALS.RECOMMENDER_NAME)
rec.append(IALS)

# RP3beta
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
							'normalize_similarity': True, 'implicit': True}
RP3beta = RP3betaRecommender(URM_train=urm_train)
RP3beta.fit(**best_hyperparams_RP3beta)
rec_names.append(RP3beta.RECOMMENDER_NAME)
rec.append(RP3beta)

# MultVAE
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask

best_hyperparams_MultVAE = {'epochs': 300, 'learning_rate': 0.0001, 'l2_reg': 3.2983231409888774e-06,
							'dropout': 0.6, 'total_anneal_steps': 100000, 'anneal_cap': 0.6, 'batch_size': 512,
							'encoding_size': 243, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 1,
							'max_parameters': 1750000000.0}
MultVAE = MultVAERecommender_OptimizerMask(URM_train=urm_train)
MultVAE.load_model(folder_path="../LoadModels80/", file_name=MultVAE.RECOMMENDER_NAME + "_best_80_new")
rec_names.append(MultVAE.RECOMMENDER_NAME)
rec.append(MultVAE)

# UserKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

best_hyperparams_UserKNNCF = {'topK': 759, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
							  'feature_weighting': 'TF-IDF'}
userknncf_cosine = UserKNNCFRecommender(URM_train=urm_train)
userknncf_cosine.fit(**best_hyperparams_UserKNNCF)
rec_names.append(userknncf_cosine.RECOMMENDER_NAME)
rec.append(userknncf_cosine)


MAP_recommender_per_group = {}

collaborative_recommender_class = {
	rec_names[0]: rec[0],
	rec_names[1]: rec[1],
	rec_names[2]: rec[2],
	rec_names[3]: rec[3],
	rec_names[4]: rec[4],
	rec_names[5]: rec[5],
	rec_names[6]: rec[6],
}

n = [0, 20, 25, 30, 35, 40, 45, 50, 58, 75, 85, 100, 120, 140, 170, 210, 270, 350, 500, 1000]
m = [20, 25, 30, 35, 40, 45, 50, 58, 75, 85, 100, 120, 140, 170, 210, 270, 350, 500, 1000, 8000]

for i in range(len(n)):
	outside, inside = data_loader.get_user_outside_n_m_interaction(n[i], m[i])
	evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], ignore_users=outside)

	for label, recommender in collaborative_recommender_class.items():
		result_df, _ = evaluator_validation.evaluateRecommender(recommender)

		if label in MAP_recommender_per_group:
			MAP_recommender_per_group[label].append(result_df.loc[10]["MAP"])
		else:
			MAP_recommender_per_group[label] = [result_df.loc[10]["MAP"]]
maps = []
for i in range(len(rec_names)):
	maps.append(MAP_recommender_per_group[rec_names[i]])

data_loader.plot_map_graph(rec_names, maps)
'''
Ranges:
1) 0-20: 3874
2) 20-25: 3862
3) 25-30: 3286
4) 30-35: 2586
5) 35-40: 2091
6) 40-45: 1783
7) 45-58: 1470
8) 50-58: 2090
9) 58-75: 3293
10) 75-85: 1564
11) 85-100: 1873
12) 100-120: 1923
13) 120-140:1531
14) 140-170: 1679
15) 170-210: 1661
16) 210-270: 1767
17) 270-350: 1513
17) 350-500: 1579
18) 500-1000: 1560
19) 1000-8000: 649
'''