from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
	split_train_in_two_percentage_global_sample
import scipy.sparse as sps
import numpy as np

urm_train_scored = sps.load_npz('../LoadModels80/URM_train_new.npz')
urm_train = sps.load_npz('../LoadModels80/URM_train_imp_new.npz')
urm_test = sps.load_npz('../LoadModels80/URM_test_new.npz')
evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], verbose=True)



# SLIMElasticNetRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
SLIMElasticNet = SLIMElasticNetRecommender(URM_train=urm_train_scored)
SLIMElasticNet.load_model(folder_path="../LoadModels80/",
						  file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_80_new.zip")

# EASE_R_Recommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
EASE_R = EASE_R_Recommender(URM_train=urm_train_scored)
EASE_R.load_model(folder_path="../LoadModels80/", file_name=EASE_R.RECOMMENDER_NAME + "_best_80_new")


# IALS
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
IALS = IALSRecommender(URM_train=urm_train_scored)
IALS.load_model(folder_path="../LoadModels80/", file_name=IALS.RECOMMENDER_NAME + "_best_80_new")

rec = [SLIMElasticNet, EASE_R, IALS]
##########################################################


import pandas as pd
import datetime
import csv
from tqdm import tqdm


def recommend(recommenders, userID):
	items = []

	rec1 = recommenders[0].recommend(userID, 10)
	print(rec1)
	rec2 = recommenders[1].recommend(userID, 10)
	print(rec2)
	rec3 = recommenders[2].recommend(userID, 10)
	print(rec3)

	for i in range(len(rec1)):
		if rec1[i] == rec2[i]:
			items.append(rec1[i])
		elif rec2[i] == rec3[i]:
			items.append(rec2[i])
		elif rec1[i] == rec3[i]:
			items.append(rec3[i])
		else:
			if rec1[i] not in items:
				items.append(rec1[i])
			elif rec2[i] not in items:
				items.append(rec2[i])
			elif rec3[i] not in items:
				items.append(rec3[i])
			else:
				items.append(np.mean([rec1[i], rec2[i], rec3[i]]).round())


	return items


def write_submission(recommender: list):

	targetUsers = pd.read_csv('../data/data_target_users_test.csv')['user_id']

	targetUsers = targetUsers.tolist()


	with open("../submissions/submission_" + datetime.datetime.now().strftime("%H_%M") + ".csv", 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['user_id', 'item_list'])

		for userID in tqdm(targetUsers):
			writer.writerow([userID, str(np.array(recommend(recommender, userID)))[1:-1]])

	print("Printing finished")


write_submission(rec)
