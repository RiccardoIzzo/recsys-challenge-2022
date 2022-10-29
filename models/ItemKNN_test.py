from Utils.load_data_old import read_data
from Utils.evaluation_n_metrics import evaluate
from Utils.prep_sub import write_csv
from Utils.preproc_n_split import preproc_n_split

import numpy as np
import scipy.sparse as sp
from similarity_functions.cosine_similarity_SLOW import cosine_similarity
from typing import Optional


# Hyperparameter
test_split = 0.2
validation_split = 0.1
shrink = 50
slice_size = 100

iai, n_episode_list, genre_list = read_data()

iai, num_users, num_items, urm_train, urm_validation, urm_test = preproc_n_split(iai,
																				 test_split=test_split,
																				 val_split=validation_split)

# ##############################################	recommender part	 ############################################# #


class CFItemKNN(object):
	def __init__(self, shrink: int):
		self.shrink = shrink
		self.weights = None

	def fit(self, urm_train: sp.csc_matrix, similarity_function):
		if not sp.isspmatrix_csc(urm_train):
			raise TypeError(f"We expected a CSC matrix, we got {type(urm_train)}")

		self.weights = similarity_function(urm_train, self.shrink)

	def recommend(self, user_id: int, urm_train: sp.csr_matrix, at: Optional[int] = None, remove_seen: bool = True):
		user_profile = urm_train[user_id]
		ranking = user_profile.dot(self.weights).A.flatten()

		if remove_seen:
			user_profile_start = urm_train.indptr[user_id]
			user_profile_end = urm_train.indptr[user_id + 1]
			seen_items = urm_train.indices[user_profile_start:user_profile_end]
			ranking[seen_items] = -np.inf

		ranking = np.flip(np.argsort(ranking))
		return ranking[:at]


recommender = CFItemKNN(shrink=shrink)

recommender.fit(urm_train.tocsc(), cosine_similarity)

########################################################################################################################

accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluate(recommender,
																						   urm_train,
																						   urm_test)

write_csv(iai, urm_train, urm_validation, recommender)
