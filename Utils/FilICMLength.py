import numpy as np
import pandas as pd
from tqdm import tqdm, trange


def comp_fav_type(user, urm, icm):
	user_item = urm.query('user_id == @user and data == 0').groupby(['item_id']).count().sort_values('user_id', ascending=False)

	for i in range(len(user_item)):
		item = user_item.iloc[i].name
		if not icm.query('item_id==@item').empty:
			return icm.query('item_id==@item')['feature_id'].item()

	return 0


def comp_type(item_number, urm, icm):
	item_list = urm.query('item_id == @item_number and data == 0').groupby(['user_id']).count().sort_values('item_id', ascending=False)
	user = item_list.iloc[0].name

	type = comp_fav_type(user, urm, icm)

	return type


def fill_t(icm, urm):
	l = []
	c = 0
	n_items = np.max(icm.item_id) + 1

	for i in trange(n_items):

		if icm['item_id'][c] == i:
			l.append([icm['item_id'][c], icm['feature_id'][c], icm['data'][c]])
			c += 1
		else:
			l.append([i, comp_type(i, urm, icm), 1.0])
			# l.append([i, 0, 0])

	return l






def fill_l(icm, urm):
	l = []
	c = 0
	n_items = np.max(icm.item_id) + 1

	for i in trange(n_items):
		if icm['item_id'][c] == i:
			l.append([icm['item_id'][c], icm['feature_id'][c], icm['data'][c]])
			c += 1
		else:
			# l.append([i, 0, calc_max_ep(i, urm)])
			l.append([i, 0, 1])

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

	return np.max(max).round()


def check_corr(icm):
	c = 0
	n_items = np.max(icm.item_id)
	for i in trange(n_items):
		if icm['item_id'][c] == i:
			c += 1
		else:
			print(i, ' :Missing')
	if c == n_items:
		return True
	return False


import numpy as np
import pandas as pd
from tqdm import tqdm, trange

urm = pd.read_csv(filepath_or_buffer='../data/interactions_and_impressions.csv',
				  dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
urm.rename(columns={urm.columns[0]: 'user_id',
					urm.columns[1]: 'item_id',
					urm.columns[2]: 'impressions',
					urm.columns[3]: 'data'},
		   inplace=True)
urm['impressions'] = urm['impressions'].replace([np.nan], '0')

ICM_length_path = '../data/data_ICM_length.csv'
icm_length = pd.read_csv(filepath_or_buffer=ICM_length_path, engine='python')
ICM_type_path = '../data/data_ICM_type.csv'
icm_type = pd.read_csv(filepath_or_buffer=ICM_type_path, engine='python')

np.mean(icm_length['data'])

new_icm_length = fill_l(icm_length, urm)
new_icm_length = pd.DataFrame(new_icm_length, columns=['item_id', 'feature_id', 'data'])
new_icm_length.to_csv('../Complete_ICM_length_all_1.csv', index=False)


new_icm_type = fill_t(icm_type, urm)
new_icm_type = pd.DataFrame(new_icm_type, columns=['item_id', 'feature_id', 'data'])
new_icm_type.to_csv('../Complete_ICM_type.csv', index=False)



