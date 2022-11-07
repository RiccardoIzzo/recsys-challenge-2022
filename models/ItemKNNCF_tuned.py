from Utils.DataLoaderSplit import DataLoaderSplit
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Utils.prep_sub import write_submission

best_hyperparams_ItemKNNCF = {'topK': 595, 'shrink': 800, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}

data_loader = DataLoaderSplit(urm='LastURM.csv')
URM, ICM_length_csr, ICM_type_csr = data_loader.get_csr_matrices()
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.85)

recommender = ItemKNNCFRecommender(URM_train=URM_train)
recommender.fit(**best_hyperparams_ItemKNNCF)

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
print(result_dict.MAP)  # 0.027

# recommender.save_model("../trained_models/", file_name=recommender.RECOMMENDER_NAME + "_best.zip")
# recommender.load_model(folder_path = "../trained_models/", file_name = recommender.RECOMMENDER_NAME + "_best.zip")
# write_submission(recommender=recommender)
