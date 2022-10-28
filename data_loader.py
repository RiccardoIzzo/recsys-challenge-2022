import pandas as pd
import numpy as np

def read_data():
	ratings = pd.read_csv('data/interactions_and_impressions.csv')
	ratings['impression_list'] = ratings['impression_list'].replace([np.nan], '0')

	return print(1+2)
