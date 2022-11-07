import numpy as np
import pandas as pd


def do_scoring():
	urm = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
					  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
	urm.rename(columns={urm.columns[0]: 'user_id',
						urm.columns[1]: 'item_id',
						urm.columns[2]: 'impressions',
						urm.columns[3]: 'data'},
			   inplace=True)
	urm['impressions'] = urm['impressions'].replace([np.nan], '0')

	ICM_length_path = '../data/Complete_ICM_length.csv'
	icm_l = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')
	ICM_type_path = '../data/Complete_ICM_type.csv'
	icm_t = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')

	new_df = create_new_df(urm, icm_l, icm_t)

	new_df.to_csv('../Master_df.csv', index=False)

	return new_df


def _avg_episode_watched_by_user(item_list, unique_items):
	n_0 = 0
	counter = 0
	for item in unique_items:
		n_0 += np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 0)
		counter += 1

	return n_0 / counter


def ret_impression(urm, user):
	impressions = []

	i_temp = urm.loc[urm['user_id'] == user, 'impressions']

	for str in i_temp:
		impressions += [int(item) for item in str.split(',') if item.isdigit()]

	return list(set(impressions))

def find_fav_type(user, urm, icm):
	user_item = urm.query('user_id == @user and data == 0').groupby(['item_id']).count().sort_values('user_id', ascending=False)

	for i in range(len(user_item)):
		item = user_item.iloc[i].name
		if not icm.query('item_id==@item').empty:
			return icm.query('item_id==@item')['feature_id'].item()

	return 0


def create_new_df(urm, icm_l, icm_t):

	from tqdm import tqdm
	df = []
	unique_users = urm.user_id.unique()

	for user in tqdm(unique_users):

		item_list = urm.loc[urm['user_id'] == user, ['item_id', 'data']]
		# item_list = urm.query('user_id == @user')



		unique_items = item_list.item_id.unique()

		avg_episode_watched_by_user = _avg_episode_watched_by_user(item_list, unique_items)

		fav_type = find_fav_type(user, urm, icm_t)

		for item in unique_items:
			n_0 = np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 0)
			n_1 = np.sum(item_list.loc[item_list['item_id'] == item, 'data'] == 1)

			impressions = ret_impression(urm, user, item)

			tot_ep = icm_l.loc[icm_l['item_id'] == item, 'data'].item()
			type = icm_t.loc[icm_t['item_id'] == item, 'feature_id']
			type = type.item() if not type.empty else 0

			df.append([user, item, n_0, n_1, avg_episode_watched_by_user, tot_ep, type, impressions, fav_type])

	return pd.DataFrame(df, columns=['user_id', 'item_id', 'n_0', 'n_1', 'avg_watched_ep', 'ep_tot', 'type', 'impressions', 'fav_type'])
#
# def maybe_better():
# 	urm = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
# 					  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
# 	urm.rename(columns={urm.columns[0]: 'user_id',
# 						urm.columns[1]: 'item_id',
# 						urm.columns[2]: 'impressions',
# 						urm.columns[3]: 'data'},
# 			   inplace=True)
# 	urm['impressions'] = urm['impressions'].replace([np.nan], '0')
#
# 	ICM_length_path = '../data/Complete_ICM_length.csv'
# 	icm_l = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')
# 	ICM_type_path = '../data/data_ICM_type.csv'
# 	icm_t = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
#
# 	grouped_df = urm.groupby(['user_id', 'item_id', 'data']).count()
# 	df = pd.merge(grouped_df[['user_id', 'item_id', 'data']], icm_l[['item_id', 'data']], how='inner', on=['item_id'])
#
