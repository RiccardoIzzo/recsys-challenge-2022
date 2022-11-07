from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Hybrid
from Utils.DataLoaderSplit import DataLoaderSplit
from Utils.prep_sub import write_submission
from Evaluation.Evaluator import EvaluatorHoldout

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

dataReader = DataLoaderSplit(urm='LastURM.csv')
URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

ItemKNNCF_recommender = ItemKNNCFRecommender(URM_train=URM_train)
ItemKNNCF_recommender.load_model(folder_path="../trained_models/",
                                 file_name=ItemKNNCF_recommender.RECOMMENDER_NAME + "_cosine_best_model_last.zip")
result_dict, _ = evaluator_test.evaluateRecommender(ItemKNNCF_recommender)
print(result_dict.MAP)

# from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
#
# EASE_R_recommender = EASE_R_Recommender(URM_train=URM_train)
# EASE_R_recommender.load_model(folder_path="../trained_models/",
#                               file_name=EASE_R_recommender.RECOMMENDER_NAME + "_best_model_last.zip")
# result_dict, _ = evaluator_test.evaluateRecommender(EASE_R_recommender)
# print(result_dict.MAP)

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender

P3alpha_recommender = P3alphaRecommender(URM_train=URM_train)
P3alpha_recommender.load_model(folder_path="../trained_models/",
                               file_name=P3alpha_recommender.RECOMMENDER_NAME + "_best_model_last.zip")
result_dict, _ = evaluator_test.evaluateRecommender(P3alpha_recommender)
print(result_dict.MAP)

from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

hybrid = ItemKNNCustomSimilarityRecommender(URM_train)
alpha = 0.3
new_similarity = (1 - alpha) * ItemKNNCF_recommender.W_sparse + alpha * P3alpha_recommender.W_sparse
hybrid.fit(new_similarity)
result_df, _ = evaluator_test.evaluateRecommender(hybrid)
print(result_df.loc[10])
