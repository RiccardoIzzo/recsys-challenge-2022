import pandas as pd
import numpy as np


def read_data():
	iai = pd.read_csv('../data/interactions_and_impressions.csv')
	iai['impression_list'] = iai['impression_list'].replace([np.nan], '0')

	n_episode_list = pd.read_csv('../data/data_ICM_length.csv')
	genre_list = pd.read_csv('../data/data_ICM_type.csv')

	return iai, n_episode_list, genre_list
