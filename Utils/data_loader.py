import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import hstack

n_users = 41629
n_items = 24507
n_item_non_preproc= 27968
n_type = 8

def preprocess_data_iai(ratings: pd.DataFrame):
	unique_users = ratings.UserID.unique()
	unique_items = ratings.ItemID.unique()

	num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
	num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

	print(num_users, min_user_id, max_user_id)
	print(num_items, min_item_id, max_item_id)

	mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "UserID": unique_users})
	mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "ItemID": unique_items})

	ratings = pd.merge(left=ratings,
					   right=mapping_user_id,
					   how="inner",
					   on="UserID")

	ratings = pd.merge(left=ratings,
					   right=mapping_item_id,
					   how="inner",
					   on="ItemID")

	return ratings


def preprocess_data_iai(matrix: pd.DataFrame):
	unique_items = matrix.ItemID.unique()

	num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

	print(num_items, min_item_id, max_item_id)

	mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

	ratings = pd.merge(left=matrix,
					   right=mapping_item_id,
					   how="inner",
					   on="ItemID")

	return ratings


class data_loader:
	def __int__(self, dataset_dir: str = "data"):
		self.n_users = n_users
		self.n_items = n_items
		self.n_item_non_preproc = n_item_non_preproc
		self.n_type = n_type

		iai = pd.read_csv('../' + dataset_dir + '/interactions_and_impressions.csv')
		iai['Impressions'] = iai['Impressions'].replace([np.nan], '0')

		ICM_length = pd.read_csv('../' + dataset_dir + '/data_ICM_length.csv')
		ICM_type = pd.read_csv('../' + dataset_dir + '/data_ICM_type.csv')

		iai = preprocess_data_iai(iai)
		ICM_length = preprocess_data(ICM_length)
		ICM_type = preprocess_data(ICM_type)

		# csv to coo sparse matrices
		self.iai_coo = coo_matrix((iai['Data'], (iai['mapped_user_id'], iai['mapped_item_id'])), shape=(n_users, n_items))
		self.ICM_length_coo = coo_matrix((ICM_length['data'], (ICM_length['item_id'], ICM_length['feature_id'])),
										 shape=(n_item_non_preproc, 1))
		self.ICM_type_coo = coo_matrix((ICM_type['data'], (ICM_type['item_id'], ICM_type['feature_id'])),
									   shape=(n_item_non_preproc, n_type))

		#coo to csr sparse matrices
		self.iai_csr = self.iai_coo.tocsr()
		self.ICM_length_csr = self.ICM_length_coo.tocsr()
		self.ICM_type_csr = self.ICM_type_coo.tocsr()

		self.episodes_per_item = np.sum(self.ICM_length_csr, axis=1)
		self.episodes_per_item_normalized = self.episodes_per_item / np.max(self.episodes_per_item)

		self.ICM_matrix = hstack(
			(self.ICM_type_csr, self.ICM_length_csr, self.episodes_per_item)).tocsr()

		self.interactions_per_user = np.ediff1d(self.iai_csr.indptr)
		self.episodes_per_item = np.ediff1d(self.ICM_length_csr.indptr)
		self.type_per_item = np.ediff1d(self.ICM_type_csr.indptr)
		self.items_per_type = np.ediff1d(self.ICM_type_csr.tocsc().indptr)

		self.users_to_recommend = pd.read_csv('../' + dataset_dir + 'data_target_users_test.csv')['UserID'].values.tolist()


	def get_coo_matrices(self):
		return self.iai_coo, self.ICM_length_coo, self.ICM_type_coo

	def get_csr_matrices(self):
		return self.iai_csr, self.ICM_length_csr, self.ICM_type_csr

	def get_targets(self):
		return self.users_to_recommend

	def get_interactions_csr(self):
		return self.iai_csr

	def get_users_under_n_interactions(self, n):
		return np.argwhere(self.interactions_per_user < n)

	def get_users_between_interactions(self, min, max):
		return np.intersect1d(np.argwhere(self.interactions_per_user >= min),
							  np.argwhere(self.interactions_per_user < max))



