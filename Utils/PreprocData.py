import pandas as pd
import numpy as np


def _preprocess_data(matrix: pd.DataFrame):

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


def _first_factor(avg_episode_watched_by_user, length, item, urm):
	x = length.loc[length['item_id'] == item, 'data'].item()
	return avg_episode_watched_by_user / x


def _second_factor(length, item, n_0):
	x = length.loc[length['item_id'] == item, 'data'].item()
	return n_0 / x


def _avg_episode_watched_by_user(item_list, unique_items):
	n_0 = 0
	counter = 0
	for item in unique_items:
		n_0 += np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 0)
		counter += 1

	return n_0 / counter


def _preprocess_df(urm, length, type):

	weight_0 = 0.8
	weight_1 = 0.2

	df = []
	unique_users = urm.user_id.unique()

	from tqdm import tqdm

	for user in tqdm(unique_users):

		item_list = urm.loc[urm['user_id'] == user, ['item_id', 'data']]
		# item_list = urm.query('user_id == @user')

		unique_items = item_list.item_id.unique()

		avg_episode_watched_by_user = _avg_episode_watched_by_user(item_list, unique_items)

		for item in unique_items:
			n_0 = np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 0)
			n_1 = np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 1)

			score = (weight_1 * n_1 * _first_factor(avg_episode_watched_by_user, length, item, urm)) + \
					(weight_0 * n_0 * _second_factor(length, item, n_0))

			df.append([user, item, score])

	return pd.DataFrame(df, columns=['user_id', 'item_id', 'data'])
# URM_df = _preprocess_df(URM, ICM_length, ICM_type)


URM = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
				  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
URM.rename(columns={URM.columns[0]: 'user_id',
					URM.columns[1]: 'item_id',
					URM.columns[2]: 'impressions',
					URM.columns[3]: 'data'},
		   inplace=True)
URM['impressions'] = URM['impressions'].replace([np.nan], '0')

ICM_length_path = '../data/data_ICM_length.csv'
ICM_type_path = '../data/data_ICM_type.csv'
ICM_type = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
ICM_length = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')

# URM = _preprocess_data(URM)

print('Start prep')
URM_df = _preprocess_df(URM, ICM_length, ICM_type)
# URM_df = URM.apply(prep(ICM_length, URM))

URM_df.to_csv('../dataframe11.csv', index=False)
