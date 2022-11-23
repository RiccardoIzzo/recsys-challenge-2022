import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack

n_users = 41629
n_items = 27968
n_item_non_preproc = 27968
n_type = 8

class DataLoaderSplit1:
	def __init__(self, dataset_dir: str = "data", test_split=0.2, val_split=0.1, master_df='master_df_df.csv'):
		self.n_users = n_users
		self.n_items = n_items
		self.n_item_non_preproc = n_item_non_preproc
		self.n_type = n_type
		self.w_0 = 0.2
		self.w_1 = 0.8

		master_df = pd.read_csv('../../data/Master_df.csv', engine='python')

		ICM_length_path = '../../data/Complete_ICM_length_OLD.csv'
		ICM_type_path = '../../data/Complete_ICM_type_OLD.csv'
		icm_t = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
		icm_l = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')

		self.master_df_df = master_df

		self.master_df_1 = pd.DataFrame(master_df.query('n_1 >= 1'), columns=['user_id', 'item_id', 'n_1']). \
			rename(columns={'n_1': 'data'})

		self.master_df_0 = pd.DataFrame(master_df.query('n_0 >= 1'), columns=['user_id', 'item_id', 'n_0'])\
			.rename(columns={'n_0': 'data'})


		# csv to coo sparse matrices
		master_df_coo0 = coo_matrix((self.master_df_0['data'], (self.master_df_0['user_id'], self.master_df_0['item_id'])), shape=(n_users, n_items))
		master_df_coo1 = coo_matrix((self.master_df_1['data'], (self.master_df_1['user_id'], self.master_df_1['item_id'])), shape=(n_users, n_items))

		# self.master_df_coo_training, self.master_df_coo_validation, self.master_df_coo_test = _dataset_splits(master_df, test_split, val_split)

		self.ICM_length_coo = coo_matrix((icm_l['data'], (icm_l['item_id'], icm_l['feature_id'])),
										 shape=(n_item_non_preproc, 1))
		self.ICM_type_coo = coo_matrix((icm_t['data'], (icm_t['item_id'], icm_t['feature_id'])),
									   shape=(n_item_non_preproc, n_type))

		#coo to csr sparse matrices
		self.master_df_csr1 = master_df_coo1.tocsr()
		self.master_df_csr0 = master_df_coo0.tocsr()

		# print('Doing Normalization')
		# from sklearn.preprocessing import normalize
		# self.normalize_master_df_csr1 = normalize(self.master_df_csr1, norm='l2', axis=1)
		# self.normalize_master_df_csr0 = normalize(self.master_df_csr0, norm='l2', axis=1)

		# self.normalize_master_df_csr = _normalize_matrix(self.master_df_csr)

		# self.master_df_csr_training = self.master_df_coo_training.tocsr()
		# self.master_df_csr_validation = self.master_df_coo_validation.tocsr()
		# self.master_df_csr_test = self.master_df_coo_test.tocsr()

		self.ICM_length_csr = self.ICM_length_coo.tocsr()
		self.ICM_type_csr = self.ICM_type_coo.tocsr()

		self.episodes_per_item = np.sum(self.ICM_length_csr, axis=1)
		self.episodes_per_item_normalized = self.episodes_per_item / np.max(self.episodes_per_item)

		self.ICM_matrix = hstack((self.ICM_type_csr, self.ICM_length_csr, self.episodes_per_item)).tocsr()

		# self.interactions_per_user = np.ediff1d(self.master_df_csr.indptr)
		# self.episodes_per_item = np.ediff1d(self.ICM_length_csr.indptr)
		# self.type_per_item = np.ediff1d(self.ICM_type_csr.indptr)
		# self.items_per_type = np.ediff1d(self.ICM_type_csr.tocsc().indptr)

		self.users_to_recommend = pd.read_csv('../' + dataset_dir + '/data_target_users_test.csv')['user_id'].values.tolist()

	# CSR Matrices
	def get_csr_matrices(self):
		return self.master_df_csr0, self.master_df_csr1, self.ICM_length_csr, self.ICM_type_csr

	def get_csr0(self):
		return self.master_df_csr0

	def get_csr1(self):
		return self.master_df_csr1

	def get_implicit_master_dfs(self):
		master_df0 = self.master_df_csr0
		master_df1 = self.master_df_csr1

		for i in range(len(self.master_df_csr0.data)):
			master_df0.data[i] = 1

		for i in range(len(self.master_df_csr1.data)):
			master_df1.data[i] = 1

		return master_df0, master_df1

	def get_users_to_recommend(self):
		return self.users_to_recommend

