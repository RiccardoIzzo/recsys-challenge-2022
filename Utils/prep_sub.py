import pandas as pd
import numpy as np
import scipy.sparse as sp
import datetime


def prepare_submission(ratings: pd.DataFrame, users_to_recommend: np.array, urm_train: sp.csr_matrix,
					   recommender: object):
	users_ids_and_mappings = ratings[ratings.UserID.isin(users_to_recommend)][["UserID", "mapped_user_id"]]\
		.drop_duplicates()
	items_ids_and_mappings = ratings[["ItemID", "mapped_item_id"]].drop_duplicates()

	mapping_to_item_id = dict(zip(ratings.mapped_item_id, ratings.ItemID))

	recommendation_length = 10
	submission = []
	for idx, row in users_ids_and_mappings.iterrows():
		UserID = row.UserID
		mapped_user_id = row.mapped_user_id

		recommendations = recommender.recommend(user_id=mapped_user_id,
												urm_train=urm_train,
												at=recommendation_length,
												remove_seen=True)

		submission.append((UserID, [mapping_to_item_id[ItemID] for ItemID in recommendations]))

	return submission


def write_submission(submissions):
	with open("../submissions/submission_" + datetime.datetime.now().strftime("%H_%M") + ".csv", "w") as f:
		f.write(f"user_id,item_list\n")
		for UserID, items in submissions:
			f.write(f"{UserID},{' '.join([str(item) for item in items])}\n")


def write_csv(iai, urm_train, urm_validation, recommender):
	"""
	:param iai:
	:param urm_train:
	:param urm_validation:
	:param recommender:
	:return: void
	"""

	users_to_recommend = pd.read_csv('../data/data_target_users_test.csv')
	users_to_recommend = np.array(users_to_recommend).reshape(1, len(users_to_recommend))


	urm_train_validation = urm_train + urm_validation

	submission = prepare_submission(iai, users_to_recommend, urm_train_validation, recommender)

	write_submission(submission)

	print("Printing finished")