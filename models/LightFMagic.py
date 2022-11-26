import scipy as sp

from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import numpy as np
from tqdm import trange


def main():
	data_loader = DataLoaderSplit(urm='URM_df.csv')

	# URM, ICM_length, ICM_type = data_loader.get_csr_matrices()
	#
	# ICMs = sp.sparse.hstack([ICM_length, ICM_type])
	# URM = sp.sparse.vstack([URM, ICMs.T])
	# URM = sp.sparse.csr_matrix(URM)
	#
	# for i in range(len(URM.data)):
	# 	if URM.data[i] < 0.1:
	# 		URM.data[i] = -1
	# 	else:
	# 		URM.data[i] = 1

	urm = data_loader.get_light_matrix()

	URM, URM_test = split_train_in_two_percentage_global_sample(urm, train_percentage=0.80)

	from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender

	recommender = LightFMCFRecommender(URM)

	hyper = {'epochs': 10,
			 'n_components': 129,
			 'loss': 'warp-kos',
			 'sgd_mode': 'adadelta',
			 'learning_rate': 0.001,
			 'item_alpha': 1.6253187239674925e-05,
			 'user_alpha': 7.20686881545928e-06}

	recommender.fit(**hyper)

	# from Utils.prep_sub import write_submission
	# write_submission(recommender)

	from Evaluation.Evaluator import EvaluatorHoldout

	evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

	result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
	print(result_dict.MAP)

if __name__ == '__main__':
	main()