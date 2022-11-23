import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack

n_users = 41629
n_items = 27968
n_item_non_preproc = 27968
n_type = 8

class DataLoaderSplit1:
	def __init__(self, dataset_dir: str = "data", test_split=0.2, val_split=0.1, urm='URM_df.csv'):
		self.n_users = n_users
		self.n_items = n_items
		self.n_item_non_preproc = n_item_non_preproc
		self.n_type = n_type
		self.w_0 = 0.2
		self.w_1 = 0.8

		urm = pd.read_csv('../data/Master_df.csv', engine='python')

		ICM_length_path = '../data/Complete_ICM_length_OLD.csv'
		ICM_type_path = '../data/Complete_ICM_type_OLD.csv'
		icm_t = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
		icm_l = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')

		self.URM_df = urm

		self.urm_1 = pd.DataFrame(urm.query('n_1 >= 1'), columns=['user_id', 'item_id', 'n_1']). \
			rename(columns={'n_1': 'data'})

		self.urm_0 = pd.DataFrame(urm.query('n_0 >= 1'), columns=['user_id', 'item_id', 'n_0'])\
			.rename(columns={'n_0': 'data'})


		# csv to coo sparse matrices
		URM_coo0 = coo_matrix((self.urm_0['data'], (self.urm_0['user_id'], self.urm_0['item_id'])), shape=(n_users, n_items))
		URM_coo1 = coo_matrix((self.urm_1['data'], (self.urm_1['user_id'], self.urm_1['item_id'])), shape=(n_users, n_items))

		# self.URM_coo_training, self.URM_coo_validation, self.URM_coo_test = _dataset_splits(URM, test_split, val_split)

		self.ICM_length_coo = coo_matrix((icm_l['data'], (icm_l['item_id'], icm_l['feature_id'])),
										 shape=(n_item_non_preproc, 1))
		self.ICM_type_coo = coo_matrix((icm_t['data'], (icm_t['item_id'], icm_t['feature_id'])),
									   shape=(n_item_non_preproc, n_type))

		#coo to csr sparse matrices
		self.URM_csr1 = URM_coo1.tocsr()
		self.URM_csr0 = URM_coo0.tocsr()

		# print('Doing Normalization')
		# from sklearn.preprocessing import normalize
		# self.normalize_URM_csr1 = normalize(self.URM_csr1, norm='l2', axis=1)
		# self.normalize_URM_csr0 = normalize(self.URM_csr0, norm='l2', axis=1)

		# self.normalize_URM_csr = _normalize_matrix(self.URM_csr)

		# self.URM_csr_training = self.URM_coo_training.tocsr()
		# self.URM_csr_validation = self.URM_coo_validation.tocsr()
		# self.URM_csr_test = self.URM_coo_test.tocsr()

		self.ICM_length_csr = self.ICM_length_coo.tocsr()
		self.ICM_type_csr = self.ICM_type_coo.tocsr()

		self.episodes_per_item = np.sum(self.ICM_length_csr, axis=1)
		self.episodes_per_item_normalized = self.episodes_per_item / np.max(self.episodes_per_item)

		self.ICM_matrix = hstack((self.ICM_type_csr, self.ICM_length_csr, self.episodes_per_item)).tocsr()

		# self.interactions_per_user = np.ediff1d(self.URM_csr.indptr)
		# self.episodes_per_item = np.ediff1d(self.ICM_length_csr.indptr)
		# self.type_per_item = np.ediff1d(self.ICM_type_csr.indptr)
		# self.items_per_type = np.ediff1d(self.ICM_type_csr.tocsc().indptr)

		self.users_to_recommend = pd.read_csv('../' + dataset_dir + '/data_target_users_test.csv')['user_id'].values.tolist()

	# CSR Matrices
	def get_csr_matrices(self):
		return self.URM_csr0, self.URM_csr1, self.ICM_length_csr, self.ICM_type_csr

	def get_csr0(self):
		return self.URM_csr0

	def get_csr1(self):
		return self.URM_csr1

	def get_implicit_urms(self):
		urm0 = self.urm_0
		urm1 = self.urm_1

		for i in range(len(self.URM_csr0.data)):
			if urm0.data[i] >= 1:
				urm0.data[i] = 1
			else:
				urm0.data[i] = 0

		for i in range(len(self.URM_csr1.data)):
			if urm1.data[i] >= 1:
				urm1.data[i] = 1
			else:
				urm1.data[i] = 0

		return urm0, urm1

	def get_users_to_recommend(self):
		return self.users_to_recommend

