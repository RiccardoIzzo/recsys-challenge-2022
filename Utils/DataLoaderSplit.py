import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

n_users = 41629
n_items = 24507
n_item_non_preproc = 27968
n_type = 8

class DataLoaderSplit:
	def __init__(self, dataset_dir: str = "data", test_split=0.2, val_split=0.1, urm='URM_df.csv'):
		self.n_users = n_users
		self.n_items = n_items
		self.n_item_non_preproc = n_item_non_preproc
		self.n_type = n_type
		self.w_0 = 0.2
		self.w_1 = 0.8

		URM_df = pd.read_csv(filepath_or_buffer='../data/' + urm, dtype={0: int, 1: int, 2: float}, engine='python')

		master_df = pd.read_csv('../data/Master_df.csv', engine='python')

		ICM_length_path = '../data/Complete_ICM_length_OLD.csv'
		ICM_type_path = '../data/Complete_ICM_type.csv'
		ICM_type = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
		ICM_length = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')
		ICM_length.drop(ICM_length.index[n_items:], axis=0, inplace=True)
		ICM_type.drop(ICM_type.index[n_items:], axis=0, inplace=True)
		self.master_df = master_df

		self.implicit_urm_1_df = pd.DataFrame(master_df.query('n_1 >= 1'), columns=['user_id', 'item_id', 'n_1']). \
			rename(columns={'n_1': 'data'})

		self.implicit_urm_0_df = pd.DataFrame(master_df.query('n_0 >= 1'), columns=['user_id', 'item_id', 'n_0']) \
			.rename(columns={'n_0': 'data'})


		# implicit matrix
		# csv to coo sparse matrices
		implicit_coo0 = coo_matrix((self.implicit_urm_0_df['data'],
									(self.implicit_urm_0_df['user_id'], self.implicit_urm_0_df['item_id'])),
								   shape=(n_users, n_items))

		implicit_coo1 = coo_matrix((self.implicit_urm_1_df['data'],
									(self.implicit_urm_1_df['user_id'], self.implicit_urm_1_df['item_id'])),
								   shape=(n_users, n_items))

		# coo to csr sparse matrices
		self.implicit_csr0 = implicit_coo0.tocsr()
		self.implicit_csr1 = implicit_coo1.tocsr()

		# End implicit

		self.URM_df = URM_df

		# csv to coo sparse matrices
		self.URM_coo = coo_matrix((URM_df['data'], (URM_df['user_id'], URM_df['item_id'])), shape=(n_users, n_items))

		self.ICM_length_coo = coo_matrix((ICM_length['data'], (ICM_length['item_id'], ICM_length['feature_id'])),
										 shape=(n_items, 1))
		self.ICM_type_coo = coo_matrix((ICM_type['data'], (ICM_type['item_id'], ICM_type['feature_id'])),
									   shape=(n_items, n_type))

		#coo to csr sparse matrices
		self.URM_csr = self.URM_coo.tocsr()

		print('Doing Normalization')
		from sklearn.preprocessing import normalize
		self.normalize_URM_csr = normalize(self.URM_csr, norm='l2', axis=1)


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

		print('Finish Data Loading')

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

	def get_ICMs(self):
		return self.ICM_length_csr, self.ICM_type_csr

	# Get implicit stuff
	def get_distinct_csr_matrices(self):
		return self.implicit_csr0, self.implicit_csr1

	def get_csr0(self):
		return self.implicit_csr0

	def get_csr1(self):
		return self.implicit_csr1

	def get_separate_implicit_csr(self):
		urm0 = self.implicit_csr0.copy()
		urm1 = self.implicit_csr1.copy()

		for i in range(len(self.implicit_csr0.data)):
			urm0.data[i] = 1

		for i in range(len(self.implicit_csr1.data)):
			urm1.data[i] = 1

		return urm0, urm1

	def get_implicit_single_mixed_csr(self):
		urm = self.implicit_csr0.copy()

		for i in range(len(self.implicit_csr0.data)):
			urm.data[i] = 1

		return urm


	# Other stuff
	def get_users_to_recommend(self):
		return self.users_to_recommend

	def get_users_under_n_interactions(self, n):
		return np.argwhere(self.interactions_per_user < n)

	def get_users_between_interactions(self, min, max):
		return np.intersect1d(np.argwhere(self.interactions_per_user >= min),
							  np.argwhere(self.interactions_per_user < max))

	def get_URM_dataframe(self):
		return self.URM_df

	def get_light_matrix(self):
		scores = (self.master_df.n_0 * 0.6) + (0.4 * self.master_df.n_1)
		urm = self.URM_df.copy()
		urm['data'] = scores
		urm['ep_len'] = self.master_df.ep_tot
		urm['type'] = self.master_df.type

		URM_coo = coo_matrix((urm['ep_len'], (urm['user_id'], urm['item_id'])), shape=(n_users, n_items))
		URM_csr = URM_coo.tocsr()
		from sklearn.preprocessing import normalize
		urm_len = normalize(URM_csr, norm='l2', axis=1)

		URM_coo = coo_matrix((urm['type'], (urm['user_id'], urm['item_id'])), shape=(n_users, n_items))
		URM_csr = URM_coo.tocsr()
		from sklearn.preprocessing import normalize
		urm_type = normalize(URM_csr, norm='l2', axis=1)

		import scipy.sparse as sps

		urm = sps.hstack([self.normalize_URM_csr, urm_len, urm_type])
		urm = sps.csr_matrix(urm)

		return urm
