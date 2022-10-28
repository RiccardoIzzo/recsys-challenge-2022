import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

def preprocess_data(ratings: pd.DataFrame):
	unique_users = ratings.user_id.unique()
	unique_items = ratings.item_id.unique()

	num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
	num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

	print(num_users, min_user_id, max_user_id)
	print(num_items, min_item_id, max_item_id)

	mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
	mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

	ratings = pd.merge(left=ratings,
					   right=mapping_user_id,
					   how="inner",
					   on="user_id")

	ratings = pd.merge(left=ratings,
					   right=mapping_item_id,
					   how="inner",
					   on="item_id")

	return ratings, num_users, num_items


def dataset_splits(ratings, num_users, num_items, validation_percentage: float, testing_percentage: float):
	seed = 42

	(user_ids_training, user_ids_test,
	 item_ids_training, item_ids_test,
	 ratings_training, ratings_test) = train_test_split(ratings.mapped_user_id,
														ratings.mapped_item_id,
														ratings.data,
														test_size=testing_percentage,
														shuffle=True,
														random_state=seed)

	(user_ids_training, user_ids_validation,
	 item_ids_training, item_ids_validation,
	 ratings_training, ratings_validation) = train_test_split(user_ids_training,
															  item_ids_training,
															  ratings_training,
															  test_size=validation_percentage)

	urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
							  shape=(num_users, num_items))

	urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
								   shape=(num_users, num_items))

	urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
							 shape=(num_users, num_items))

	return urm_train, urm_validation, urm_test


def preproc_n_split(iai, test_split: float = 0.2, val_split: float = 0.1):
	"""
	:param iai:
	:param test_split:
	:param val_split:
	:return: iai, num_users, num_items, urm_train, urm_validation, urm_test
	"""
	iai, num_users, num_items = preprocess_data(iai)

	test_split = test_split
	validation_split = val_split

	urm_train, urm_validation, urm_test = dataset_splits(iai,
														 num_users=num_users,
														 num_items=num_items,
														 validation_percentage=validation_split,
														 testing_percentage=test_split)

	return iai, num_users, num_items, urm_train, urm_validation, urm_test
