import numpy as np
import scipy.sparse as sp


def recall(recommendations: np.array, relevant_items: np.array) -> float:
	is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

	recall_score = np.sum(is_relevant) / relevant_items.shape[0]

	return recall_score


def precision(recommendations: np.array, relevant_items: np.array) -> float:
	is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

	precision_score = np.sum(is_relevant) / recommendations.shape[0]

	return precision_score


def mean_average_precision(recommendations: np.array, relevant_items: np.array) -> float:
	is_relevant = np.in1d(recommendations, relevant_items, assume_unique=True)

	precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

	map_score = np.sum(precision_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

	return map_score


def evaluator(recommender: object, urm_train: sp.csr_matrix, urm_test: sp.csr_matrix):
	recommendation_length = 10
	accum_precision = 0
	accum_recall = 0
	accum_map = 0

	num_users = urm_train.shape[0]

	num_users_evaluated = 0
	num_users_skipped = 0
	for user_id in range(num_users):
		user_profile_start = urm_test.indptr[user_id]
		user_profile_end = urm_test.indptr[user_id + 1]

		relevant_items = urm_test.indices[user_profile_start:user_profile_end]

		if relevant_items.size == 0:
			num_users_skipped += 1
			continue

		recommendations = recommender.recommend(user_id=user_id,
												at=recommendation_length,
												urm_train=urm_train,
												remove_seen=True)
		accum_precision += precision(recommendations, relevant_items)
		accum_recall += recall(recommendations, relevant_items)
		accum_map += mean_average_precision(recommendations, relevant_items)

		num_users_evaluated += 1

	accum_precision /= max(num_users_evaluated, 1)
	accum_recall /= max(num_users_evaluated, 1)
	accum_map /= max(num_users_evaluated, 1)

	return accum_precision, accum_recall, accum_map, num_users_evaluated, num_users_skipped


def evaluate(recommender, urm_train, urm_test):
	"""
	:param recommender:
	:param urm_train:
	:param urm_test:
	:return: accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped
	"""
	accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluator(recommender,
																								urm_train,
																								urm_test)
	print(accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped)

	return accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped