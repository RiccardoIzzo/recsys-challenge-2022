import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

n_users = 41629
n_items = 27968
n_item_non_preproc = 27968
n_type = 8


def _preprocess_data(matrix: pd.DataFrame):

	# if matrix.columns[0] == 'user_id':
	unique_users = matrix.user_id.unique()
	unique_items = matrix.item_id.unique()

	num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
	num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

	print(num_users, min_user_id, max_user_id)
	print(num_items, min_item_id, max_item_id)

	mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
	mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

	matrix = pd.merge(left=matrix,
					  right=mapping_user_id,
					  how="inner",
					  on="user_id")

	matrix = pd.merge( left=matrix,
					   right=mapping_item_id,
					   how="inner",
					   on="item_id")
	return matrix


def _normalize_matrix(matrix):
	max = np.max(matrix.data)
	min = np.min(matrix.data)

	matrix.data = (matrix.data - min) / (max - min)

	return matrix

class DataLoaderSplit:
	def __init__(self, dataset_dir: str = "data", test_split=0.2, val_split=0.1):
		self.n_users = n_users
		self.n_items = n_items
		self.n_item_non_preproc = n_item_non_preproc
		self.n_type = n_type

		# URM = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
		# 				  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
		URM_df = pd.read_csv(filepath_or_buffer='../data/interactionScored.csv',
						  dtype={0: int, 1: int, 2: int}, engine='python')
		URM_df.rename(columns={URM_df.columns[0]: 'user_id',
							URM_df.columns[1]: 'item_id',
							# URM_df.columns[2]: 'impressions',
							URM_df.columns[2]: 'data'},
				   inplace=True)
		# URM_df['impressions'] = URM_df['impressions'].replace([np.nan], '0')

		ICM_length_path = '../data/data_ICM_length.csv'
		ICM_type_path = '../data/data_ICM_type.csv'
		ICM_type = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
		ICM_length = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')

		# print('Start prep')
		# URM_df = _preprocess_df(URM, ICM_length, ICM_type)

		# URM = _preprocess_data(URM)
		self.URM_df = URM_df
		# ICM_length = _preprocess_data(ICM_length)
		# ICM_type = _preprocess_data(ICM_type)

		# csv to coo sparse matrices
		self.URM_coo = coo_matrix((URM_df['data'], (URM_df['user_id'], URM_df['item_id'])), shape=(n_users, n_items))

		# self.URM_coo_training, self.URM_coo_validation, self.URM_coo_test = _dataset_splits(URM, test_split, val_split)

		self.ICM_length_coo = coo_matrix((ICM_length['data'], (ICM_length['item_id'], ICM_length['feature_id'])),
										 shape=(n_item_non_preproc, 1))
		self.ICM_type_coo = coo_matrix((ICM_type['data'], (ICM_type['item_id'], ICM_type['feature_id'])),
									   shape=(n_item_non_preproc, n_type))

		#coo to csr sparse matrices
		self.URM_csr = self.URM_coo.tocsr()

		print('Doing Normalization')
		from sklearn.preprocessing import normalize
		self.normalize_URM_csr = normalize(self.URM_csr, norm='l2', axis=1)

		# self.normalize_URM_csr = _normalize_matrix(self.URM_csr)

		# self.URM_csr_training = self.URM_coo_training.tocsr()
		# self.URM_csr_validation = self.URM_coo_validation.tocsr()
		# self.URM_csr_test = self.URM_coo_test.tocsr()

		self.ICM_length_csr = self.ICM_length_coo.tocsr()
		self.ICM_type_csr = self.ICM_type_coo.tocsr()

		self.episodes_per_item = np.sum(self.ICM_length_csr, axis=1)
		self.episodes_per_item_normalized = self.episodes_per_item / np.max(self.episodes_per_item)

		self.ICM_matrix = hstack((self.ICM_type_csr, self.ICM_length_csr, self.episodes_per_item)).tocsr()

		self.interactions_per_user = np.ediff1d(self.URM_csr.indptr)
		self.episodes_per_item = np.ediff1d(self.ICM_length_csr.indptr)
		self.type_per_item = np.ediff1d(self.ICM_type_csr.indptr)
		self.items_per_type = np.ediff1d(self.ICM_type_csr.tocsc().indptr)

		self.users_to_recommend = pd.read_csv('../' + dataset_dir + '/data_target_users_test.csv')['user_id'].values.tolist()

	# COO Matrices
	def get_coo_matrices(self):
		return self.URM_coo, self.ICM_length_coo, self.ICM_type_coo

	# def get_split_URM_coo_matrix(self):
	# 	return self.URM_coo_training, self.URM_coo_validation, self.URM_coo_test
	# 
	# def get_all_coo_matrices(self):
	# 	return self.URM_coo_training, self.URM_coo_validation, self.URM_coo_test, self.ICM_length_coo, self.ICM_type_coo

	# CSR Matrices
	def get_csr_matrices(self):
		return self.normalize_URM_csr, self.ICM_length_csr, self.ICM_type_csr

	# def get_split_URM_csr_matrix(self):
	# 	return self.URM_csr_training, self.URM_csr_validation, self.URM_csr_test
	# 
	# def get_all_csr_matrices(self):
	# 	return self.URM_csr_training, self.URM_csr_validation, self.URM_csr_test, self.ICM_length_csr, self.ICM_type_csr

	# Other stuff
	def get_users_to_recommend(self):
		return self.users_to_recommend

	def get_interactions_csr(self):
		return self.URM_csr_training, self.URM_csr_validation, self.URM_csr_test

	def get_users_under_n_interactions(self, n):
		return np.argwhere(self.interactions_per_user < n)

	def get_users_between_interactions(self, min, max):
		return np.intersect1d(np.argwhere(self.interactions_per_user >= min),
							  np.argwhere(self.interactions_per_user < max))

	def get_new_split_csr_matrix(self, test_split=0.2, val_split=0.1):
		URM_coo_training, URM_coo_validation, URM_coo_test = _dataset_splits(self.URM, test_split, val_split)

		return URM_coo_training.tocsr(), URM_coo_validation.tocsr(), URM_coo_test.tocsr()

	def get_URM_dataframe(self):
		return self.URM
