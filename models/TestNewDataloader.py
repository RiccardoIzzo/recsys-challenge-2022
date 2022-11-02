import numpy as np

from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

# Hyperparameter
test_split = 0.2
validation_split = 0.1

data_loader = DataLoaderSplit()
URM, n_episode_list, types = data_loader.get_csr_matrices()
# iai = data_loader.get_iai()
URM, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage = 0.80)

from Recommenders.NonPersonalizedRecommender import TopPop

best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}

recommender = TopPop(URM_train=URM)
# recommender.fit(**best_hyperparams_ItemKNNCF)
recommender.fit()

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)

write_submission(recommender)

# ######################################################################################################################
import numpy as np
from Utils.prep_sub import write_submission
from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

data_loader = DataLoaderSplit()
URM, ICM_length, ICM_type = data_loader.get_csr_matrices()

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.15)

from Recommenders.NonPersonalizedRecommender import TopPop

best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}

recommender = TopPop(URM_train=URM_train)
# recommender.fit(**best_hyperparams_ItemKNNCF)
recommender.fit()

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)

write_submission(recommender)


# ######################################################################################################################

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.ChallengeReader import ChallengeReader
# from Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
# from Data_manager.DataReaderPostprocessing_K_Cores import DataReaderPostprocessing_K_Cores
# from Data_manager.DataReaderPostprocessing_User_sample import DataReaderPostprocessing_User_sample
# from Data_manager.DataReaderPostprocessing_Implicit_URM import DataReaderPostprocessing_Implicit_URM

dataReader = ChallengeReader()
dataset = dataReader.load_data()

URM_all = dataset.get_URM_all()

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.15)

from sklearn.preprocessing import normalize
URM_all = normalize(URM_all, norm='l2', axis=1)

best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}


from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

recommender = ItemKNNCFRecommender(URM_train=URM_train)
recommender.fit(**best_hyperparams_ItemKNNCF)


# from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
#
# recommender = SLIMElasticNetRecommender(URM_train)
# recommender.fit(topK=1139, l1_ratio=6.276359878274636e-05, alpha=0.12289267654724283)

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)

from Utils.prep_sub import write_submission
write_submission(recommender)
