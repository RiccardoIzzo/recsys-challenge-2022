import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
import scipy.sparse as sps

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
		self.icm_length = ICM_length
		self.icm_type = ICM_type
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

		self.ICM_length_coo = coo_matrix((ICM_length['data'], (ICM_length['item_id'], ICM_length['data'])),
										 shape=(n_items, 2001))
		self.ICM_type_coo = coo_matrix((ICM_type['feature_id'], (ICM_type['item_id'], ICM_type['feature_id'])),
									   shape=(n_items, n_type))

		#coo to csr sparse matrices
		self.URM_csr = self.URM_coo.tocsr()

		print('Doing Normalization')
		from sklearn.preprocessing import normalize
		self.normalize_URM_csr = normalize(self.URM_csr, norm='l2', axis=1)

		# self.normalize_URM_csr = self.URM_csr
		# from sklearn.preprocessing import MinMaxScaler
		# scaler = MinMaxScaler((0,5))
		# data = scaler.fit_transform(self.URM_csr.data.reshape(-1,1))
		# self.normalize_URM_csr = self.URM_csr
		# self.normalize_URM_csr.data = data.reshape(1, -1)[0]
		# for i in range(len(self.normalize_URM_csr.data)):
		# 	if self.normalize_URM_csr.data[i] < 1:
		# 		self.normalize_URM_csr.data[i] = 0


		self.ICM_length_csr = self.ICM_length_coo.tocsr()
		self.ICM_type_csr = self.ICM_type_coo.tocsr()

		for i in range(len(self.ICM_length_csr.data)):
			self.ICM_length_csr.data[i] = 1
			self.ICM_type_csr.data[i] = 1


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
		urm = self.URM_csr.copy()

		for i in range(len(self.URM_csr.data)):
			urm.data[i] = 1

		return urm


	# Other stuff
	def get_users_to_recommend(self):
		return self.users_to_recommend

	def get_users_under_n_interactions(self, n):
		master = self.master_df.copy().groupby(['user_id', 'inter']).sum().reset_index()
		user_above = (master['user_id'] + 1) * (master['inter'] >= n)
		user_up = []
		user_down = []

		for i in range(len(user_above)):
			if user_above[i] != 0:
				user_up.append(i)
			else:
				user_down.append(i)

		return user_down, user_up

	def get_users_between_interactions(self, min, max):
		return np.intersect1d(np.argwhere(self.interactions_per_user >= min),
							  np.argwhere(self.interactions_per_user < max))

	def get_matrix_under_n_interaction(self, n, implicit=True):
		pd.set_option('mode.chained_assignment', None)
		urm1 = self.URM_df.copy()
		urm2 = self.URM_df.copy()

		urm1['score'] = self.master_df['inter']
		urm2['score'] = self.master_df['inter']

		urm1['value'] = urm1.data * (urm1.score <= n)
		urm2['value'] = urm2.data * (urm2.score > n)

		coo1 = sps.coo_matrix((urm1['value'], (urm1['user_id'], urm1['item_id'])), shape=(n_users, n_items))
		coo2 = sps.coo_matrix((urm2['value'], (urm2['user_id'], urm2['item_id'])), shape=(n_users, n_items))
		csr1 = coo1.tocsr()
		csr2 = coo2.tocsr()

		if implicit:
			for i in range(len(csr2.data)):
				if csr1.data[i] != 0:
					csr1.data[i] = 1
				else:
					csr2.data[i] = 1
			normalize_URM_csr1 = csr1
			normalize_URM_csr2 = csr2
		else:
			from sklearn.preprocessing import normalize
			normalize_URM_csr1 = normalize(csr1, norm='l2', axis=1)
			normalize_URM_csr2 = normalize(csr2, norm='l2', axis=1)

		pd.reset_option("mode.chained_assignment")
		# less interaction and most interaction
		return normalize_URM_csr1, normalize_URM_csr2

	def get_user_outside_n_m_interaction(self, n, m):
		urm1 = self.URM_df.copy()

		master11 = self.master_df.copy().groupby(['user_id', 'inter']).sum().reset_index()

		urm1['score1'] = self.master_df['inter']
		urm1['score2'] = self.master_df['inter']

		urm1['value1'] = urm1.data * (urm1.score1 >= n)
		urm1['value2'] = urm1.data * (urm1.score2 < m)
		urm1['value'] = urm1.data * (urm1.value1 * urm1.value2 != 0)

		urm11 = (urm1.groupby(['user_id']).sum().reset_index())
		list_user_inside = (master11['user_id'] * (urm11.value != 0)).tolist()
		list_user_outside = (master11['user_id'] * (urm11.value == 0)).tolist()

		list_user_inside = [i for i in list_user_inside if i != 0]

		list_user_outside = [i for i in list_user_outside if i != 0]
		if n <= 157 < m:
			list_user_inside.insert(0, 0)
		else:
			list_user_outside.insert(0, 0)

		return list_user_outside, list_user_inside

	def get_matrix_between_n_m_interaction(self, n, m, implicit=True):
		pd.set_option('mode.chained_assignment', None)

		urm1 = self.URM_df.copy()
		urm2 = self.URM_df.copy()

		master11 = self.master_df.copy().groupby(['user_id', 'inter']).sum().reset_index()

		urm1['score1'] = self.master_df['inter']
		urm1['score2'] = self.master_df['inter']
		urm2['score1'] = self.master_df['inter']
		urm2['score2'] = self.master_df['inter']

		urm1['value1'] = urm1.data * (urm1.score1 >= n)
		urm1['value2'] = urm1.data * (urm1.score2 < m)
		urm1['value'] = urm1.data * (urm1.value1 * urm1.value2 != 0)

		urm11 = (urm1.groupby(['user_id']).sum().reset_index())
		list_user_below = (master11['user_id'] * (urm11.value != 0)).tolist()
		list_user_below = [i for i in list_user_below if i != 0]
		if urm1['value'][0] != 0:
			list_user_below.insert(0, 0)

		urm2['value1'] = urm2.data * (urm2.score1 < n)
		urm2['value2'] = urm2.data * (urm2.score2 >= m)
		urm2['value'] = urm2.data * (urm2.value1 + urm2.value2 != 0)

		coo1 = sps.coo_matrix((urm1['value'], (urm1['user_id'], urm1['item_id'])), shape=(n_users, n_items))
		coo2 = sps.coo_matrix((urm2['value'], (urm2['user_id'], urm2['item_id'])), shape=(n_users, n_items))
		csr1 = coo1.tocsr()
		csr2 = coo2.tocsr()

		if implicit:
			for i in range(len(csr2.data)):
				if csr1.data[i] != 0:
					csr1.data[i] = 1
				else:
					csr2.data[i] = 1
			normalize_URM_csr1 = csr1
			normalize_URM_csr2 = csr2
		else:
			from sklearn.preprocessing import normalize
			normalize_URM_csr1 = normalize(csr1, norm='l2', axis=1)
			normalize_URM_csr2 = normalize(csr2, norm='l2', axis=1)

		pd.reset_option("mode.chained_assignment")
		# less interaction and most interaction
		return normalize_URM_csr1, normalize_URM_csr2, list_user_below

	def get_URM_dataframe(self):
		return self.URM_df

	def get_user_feature(self):

		users = self.master_df.copy()
		users = users.groupby(['user_id', 'avg_watched_ep', 'fav_type']).sum().reset_index()
		users.drop(['item_id', 'ep_tot', 'type'], axis=1, inplace=True)
		users['avg_watched_ep'] = users['avg_watched_ep'].apply(lambda x: np.round(x))

		avg_coo = sps.coo_matrix((users['avg_watched_ep'], (users['user_id'], users['avg_watched_ep'])), shape=(n_users, int(np.max(users['avg_watched_ep'])) + 1))
		fav_coo = sps.coo_matrix((users['fav_type'], (users['user_id'], users['fav_type'])), shape=(n_users, int(np.max(users['fav_type'])) + 1))
		n_0_coo = sps.coo_matrix((users['n_0'], (users['user_id'], users['n_0'])), shape=(n_users, int(np.max(users['n_0'])) + 1))
		n_1_coo = sps.coo_matrix((users['n_1'], (users['user_id'], users['n_1'])), shape=(n_users, int(np.max(users['n_1'])) + 1))

		# import pandas as pd
		# master_df = pd.read_csv('../data/Master_df.csv', engine='python')

		avg_csr = avg_coo.tocsr()
		fav_csr = fav_coo.tocsr()
		n_0_csr = n_0_coo.tocsr()
		n_1_csr = n_1_coo.tocsr()

		fav_csr.data = fav_csr.data.astype(np.float32)
		n_0_csr.data = n_0_csr.data.astype(np.float32)
		n_1_csr.data = n_1_csr.data.astype(np.float32)

		for i in range(len(avg_csr.data)):
			avg_csr.data[i] = 0.25
		for i in range(len(fav_csr.data)):
			fav_csr.data[i] = 0.25
		for i in range(len(n_0_csr.data)):
			n_0_csr.data[i] = 0.3
		for i in range(len(n_1_csr.data)):
			n_1_csr.data[i] = 0.2

		users_csr = sps.hstack([n_0_csr, n_1_csr, avg_csr, fav_csr])

		return users_csr, users

	def get_item_feature(self):
		icms = pd.concat([self.icm_length.drop('feature_id', axis=1), self.icm_type.drop(['data', 'item_id'], axis=1)],
						 axis=1, join="inner")

		icms.drop(['item_id'], axis=1, inplace=True)
		# icms = icms.div(icms.sum(axis=1), axis=0)
		icms_csr = sps.hstack([self.ICM_length_csr, self.ICM_type_csr])

		return sps.csr_matrix(icms_csr), icms

	def create_feature_matrices(self):
		ifm, idf = self.get_item_feature()
		ufm, udf = self.get_user_feature()

		id_mat = np.identity(n_items)
		id = sps.csr_matrix(id_mat)
		ifm_final = sps.hstack([id, ifm])

		id_mat_users = np.identity(n_users)
		id_users = sps.csr_matrix(id_mat_users)
		ufm_final = sps.hstack([id_users, ufm])

		return ufm_final, ifm_final

	def create_matrix_light(self):
		from sklearn.preprocessing import OneHotEncoder

		master = self.master_df.copy()
		onehot_encoder = OneHotEncoder(sparse=True)

		users_encoded = onehot_encoder.fit_transform(master['user_id'].values.reshape(-1, 1))
		item_encoded = onehot_encoder.fit_transform(master['item_id'].values.reshape(-1, 1))

		m = sps.csr_matrix(sps.hstack([users_encoded, item_encoded]))

		return m

	def plot_map_graph(self, names, maps):
		import matplotlib.pyplot as plt

		_ = plt.figure(figsize=(16, 9))
		for i in range(len(names)):
			results = maps[i]
			plt.scatter(x=np.arange(0, len(results)), y=results, label=names[i])
		plt.ylabel('MAP')
		plt.xlabel('User Group')
		plt.legend()
		plt.show()



