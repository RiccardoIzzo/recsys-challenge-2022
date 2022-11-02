from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.NonPersonalizedRecommender import *
from run_all_algorithms import _get_instance

from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Utils.prep_sub import write_submission

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.ChallengeReader import ChallengeReader

dataReader = ChallengeReader()
dataset = dataReader.load_data()

URM_all = dataset.get_URM_all()


def normalize_matrix(matrix):
	max = np.max(matrix.data)
	min = np.min(matrix.data)

	matrix.data = (matrix.data - min) / (max - min)

	return matrix


URM_all = normalize_matrix(URM_all)

# from sklearn.preprocessing import normalize
# URM_all = normalize(URM_all, norm='l2', axis=1)

ICM_length = dataset.get_ICM_from_name('ICM_length')
ICM_type = dataset.get_ICM_from_name('ICM_type')

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.85)
# URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.9)

# evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
#
# recommender_class = TopPop
# n_cases = 10
# n_random_starts = int(n_cases * 0.3)
# metric_to_optimize = "MAP"
# cutoff_to_optimize = 10
#
# runHyperparameterSearch_Collaborative(recommender_class,
#                                       URM_train=URM_train,
#                                       URM_train_last_test=URM_validation,
#                                       metric_to_optimize=metric_to_optimize,
#                                       cutoff_to_optimize=cutoff_to_optimize,
#                                       evaluator_validation_earlystopping=evaluator_validation,
#                                       evaluator_validation=evaluator_validation,
#                                       evaluator_test=evaluator_test,
#                                       output_folder_path='../trained_models',
#                                       parallelizeKNN=True,
#                                       allow_weighting=True,
#                                       resume_from_saved=True,
#                                       save_model="best",
#                                       similarity_type_list=['cosine', 'pearson', 'jaccard', 'asymmetric', 'dice', 'tversky', 'tanimoto', 'adjusted', 'euclidean'],
#                                       n_cases=n_cases,
#                                       n_random_starts=n_random_starts)
#
# recommender_object = _get_instance(recommender_class, URM_train, ICM_length, ICM_type)
# recommender_object.load_model('../trained_models', file_name="/TopPopRecommender_best_model_last.zip")
# recommender_object.fit()

recommender = TopPop(URM_train=URM_train)
recommender.fit()

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

result_dict, _ = evaluator_test.evaluateRecommender(recommender)
print(result_dict.MAP)



write_submission(recommender=recommender)

#
# from Utils.prep_sub import write_csv
# import pandas as pd
# import numpy as np
#
# URM_all_dataframe = pd.read_csv('../data/interactions_and_impressions.csv', dtype={0: str, 1: str, 2: str, 3: int}, engine='python')
# URM_all_dataframe.rename(columns={URM_all_dataframe.columns[0]: 'user_id',
# 								  URM_all_dataframe.columns[1]: 'item_id',
# 								  URM_all_dataframe.columns[2]: 'impressions',
# 								  URM_all_dataframe.columns[3]: 'data'},
# 						 inplace=True)
# URM_all_dataframe.columns = ["user_id", "item_id", "impressions", "data"]
#
#
# write_csv(URM_all_dataframe, URM_train, URM_validation, recommender)

















'''

quello che c'è su github, come backup


from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.NonPersonalizedRecommender import *
from run_all_algorithms import _get_instance

from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Utils.prep_sub import write_submission

# Hyperparameter
test_split = 0.2
validation_split = 0.1

data_loader = DataLoaderSplit()
iai = data_loader.get_iai()
train, test, val, n_episode_list, ICM_dataframe = data_loader.get_all_csr_matrices()

evaluator_validation = EvaluatorHoldout(val, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(train, cutoff_list=[10])

recommender_class = TopPop
n_cases = 10
n_random_starts = int(n_cases * 0.3)
metric_to_optimize = "MAP"
cutoff_to_optimize = 10

runHyperparameterSearch_Collaborative(recommender_class,
                                      URM_train=train,
                                      URM_train_last_test=val,
                                      metric_to_optimize=metric_to_optimize,
                                      cutoff_to_optimize=cutoff_to_optimize,
                                      evaluator_validation_earlystopping=evaluator_validation,
                                      evaluator_validation=evaluator_validation,
                                      evaluator_test=evaluator_test,
                                      output_folder_path='../trained_models',
                                      parallelizeKNN=True,
                                      allow_weighting=True,
                                      resume_from_saved=True,
                                      save_model="best",
                                      similarity_type_list=['cosine', 'pearson', 'jaccard', 'asymmetric', 'dice', 'tversky', 'tanimoto', 'adjusted', 'euclidean'],
                                      n_cases=n_cases,
                                      n_random_starts=n_random_starts)

recommender_object = _get_instance(recommender_class, train, ICM_dataframe, ICM_dataframe)  # UCM?
recommender_object.load_model('../trained_models', file_name="/TopPopRecommender_best_model_last.zip")

# recommender = TopPop(URM_train=train)
# recommender.fit()

recommender_object.fit()
result_dict, _ = evaluator_validation.evaluateRecommender(recommender_object)
print(result_dict.loc[10])

write_submission(recommender=recommender_object)
'''

