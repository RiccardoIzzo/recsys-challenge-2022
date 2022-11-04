import numpy as np
import pandas as pd
from tqdm import tqdm, trange


def fill(icm, urm):
	l = []
	c = 0
	n_items = np.max(icm.item_id) + 1

	for i in trange(n_items):
		if icm['item_id'][c] == i:
			l.append([icm['item_id'][c], icm['feature_id'][c], icm['data'][c]])
			c += 1
		else:
			l.append([i, 0, calc_max_ep(i, urm)])

	return l


def calc_max_ep(item_number, urm):
	# cerca in urm max episode di item e assegna a x
	# users_list = urm.loc[urm['item_id'] == item_number, ['user_id', 'data']]
	max = []

	if not urm.loc[urm['item_id'] == item_number].empty:
		users_list = urm.query('item_id == @item_number and data == 0')
		unique_users = users_list.user_id.unique()

		for user in unique_users:
			max.append(np.sum(users_list.loc[users_list['user_id'] == user, 'data'] == 0))

	else:
		return 0

	return np.mean(max).round()


def check_corr(icm):
	c = 0
	n_items = np.max(icm.item_id) + 1
	for i in trange(n_items):
		if icm['item_id'][c] == i:
			c += 1
		else:
			print(i, ' :Missing')
	return


urm = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
				  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
urm.rename(columns={urm.columns[0]: 'user_id',
					urm.columns[1]: 'item_id',
					urm.columns[2]: 'impressions',
					urm.columns[3]: 'data'},
		   inplace=True)
urm['impressions'] = urm['impressions'].replace([np.nan], '0')

ICM_length_path = '../data/data_ICM_length.csv'
ICM_type_path = '../data/data_ICM_type.csv'
ICM_type = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')
ICM_length = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')

new_icm_list = fill(ICM_length, urm)
# print(new_icm_list)
new_icm = pd.DataFrame(new_icm_list, columns=['item_id', 'feature_id', 'data'])

new_icm.to_csv('../Complete_ICM_length.csv', index=False)
