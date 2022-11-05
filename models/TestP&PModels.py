from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

# TopPop new urm, map: 0.012798
# TopPop old urm, map: 0.012806

# data_loader = DataLoaderSplit(urm='interactionScored.csv')
data_loader = DataLoaderSplit(urm='interactionScoredOld.csv')

URM, n_episode_list, types = data_loader.get_csr_matrices()

URM, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

from Recommenders.NonPersonalizedRecommender import TopPop

recommender = TopPop(URM_train=URM)
recommender.fit()

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)

# ######################################################################################################################

from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

# ItemCFKNN new urm, map: 0.0182
# ItemCFKNN old urm, map: 0.020445

data_loader = DataLoaderSplit(urm='interactionScored.csv')
# data_loader = DataLoaderSplit(urm='interactionScoredOld.csv')

URM, n_episode_list, types = data_loader.get_csr_matrices()

URM, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

recommender = ItemKNNCFRecommender(URM_train=URM)
recommender.fit(**best_hyperparams_ItemKNNCF)

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)

# ######################################################################################################################

from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

data_loader = DataLoaderSplit(urm='interactionScored.csv')
# data_loader = DataLoaderSplit(urm='interactionScoredOld.csv')

URM, n_episode_list, types = data_loader.get_csr_matrices()

URM, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.80)

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

recommender = EASE_R_Recommender(URM_train=URM)
recommender.fit(topK=None, l2_norm=0.005)

# from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
#
# recommender = SLIMElasticNetRecommender(URM)
# recommender.fit(topK=1139, l1_ratio=6.276359878274636e-05, alpha=0.12289267654724283)

from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])

result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)
