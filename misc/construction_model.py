from Utils.data_loader import read_data
from Utils.evaluation_n_metrics import evaluate
from Utils.prep_sub import write_csv
from Utils.preproc_n_split import preproc_n_split

import numpy as np
import scipy.sparse as sp

# Hyperparameter
test_split = 0.2
validation_split = 0.1


iai, n_episode_list, genre_list = read_data()

iai, num_users, num_items, urm_train, urm_validation, urm_test = preproc_n_split(iai,
																				 test_split=test_split,
																				 val_split=validation_split)

# ##############################################	recommender part	 ############################################# #

recommender = []

########################################################################################################################

accum_precision, accum_recall, accum_map, num_user_evaluated, num_users_skipped = evaluate(recommender,
																						   urm_train,
																						   urm_test)

write_csv(iai, urm_train, urm_validation, recommender)