from Utils.DataLoaderSplit import DataLoaderSplit
from Utils.prep_sub import write_submission
from Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
import numpy as np

dataReader = DataLoaderSplit(urm='newURM.csv')
URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
URM_all_imp = dataReader.get_implicit_single_mixed_csr()

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
best_hyperparams_ItemKNNCF_asymmetric = {'topK': 50, 'shrink': 77, 'similarity': 'asymmetric', 'normalize': True,
                                         'asymmetric_alpha': 0.3388995614775971, 'feature_weighting': 'TF-IDF'}
ItemKNNCF_asymmetric = ItemKNNCFRecommender(URM_train=URM_all)
ItemKNNCF_asymmetric.fit(**best_hyperparams_ItemKNNCF_asymmetric)

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
best_hyperparams_ItemKNNCF_tversky = {'topK': 50, 'shrink': 0, 'similarity': 'tversky', 'normalize': True,
                                      'tversky_alpha': 0.2677659937917337, 'tversky_beta': 1.8732432514946602}
ItemKNNCF_tversky = ItemKNNCFRecommender(URM_train=URM_all_imp)
ItemKNNCF_tversky.fit(**best_hyperparams_ItemKNNCF_tversky)

itemknncf_asymmetric_tversky = DifferentLossScoresHybridRecommender(URM_all_imp, ItemKNNCF_asymmetric, ItemKNNCF_tversky)
itemknncf_asymmetric_tversky.fit(1, 0.6384198928672995)

from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
best_hyperparams_UserKNNCF = {'topK': 759, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}
UserKNNCF_cosine = UserKNNCFRecommender(URM_train=URM_all_imp)
UserKNNCF_cosine.fit(**best_hyperparams_UserKNNCF)

item_user_knncf= DifferentLossScoresHybridRecommender(URM_all_imp, itemknncf_asymmetric_tversky, UserKNNCF_cosine)
item_user_knncf.fit(-np.inf, 0.7743400908473544)

from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

best_hyperparams_SLIM_BPR_Cython = {'topK': 89, 'epochs': 300, 'symmetric': True, 'sgd_mode': 'adagrad',
                                    'lambda_i': 0.001, 'lambda_j': 9.773899284052082e-05,
                                    'learning_rate': 0.03500210788755942}
SLIM_BPR_Cython = SLIM_BPR_Cython(URM_all_imp)
SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
SLIMElasticNet = SLIMElasticNetRecommender(URM_train=URM_all_imp)
SLIMElasticNet.load_model(folder_path="../trained_models/", file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_100")

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
EASE_R = EASE_R_Recommender(URM_train=URM_all_imp)
EASE_R.load_model(folder_path="../trained_models/", file_name=EASE_R.RECOMMENDER_NAME + "_best_100")

from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
IALS = IALSRecommender(URM_train=URM_all)
IALS.load_model(folder_path="../trained_models/", file_name=IALS.RECOMMENDER_NAME + "_best_100_new")

from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
MultVAE = MultVAERecommender_OptimizerMask(URM_train=URM_all_imp)
MultVAE.load_model(folder_path="../trained_models/", file_name=MultVAE.RECOMMENDER_NAME + "_best_100")

slim_vae = DifferentLossScoresHybridRecommender(URM_all_imp, SLIMElasticNet, MultVAE)
slim_vae.fit(1, 0.34597298157858547)

slim_vae_itemuserknncf = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae, item_user_knncf)
slim_vae_itemuserknncf.fit(1, 0.908754409674423)

slim_vae_itemuserknncf_ials = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_itemuserknncf, IALS)
slim_vae_itemuserknncf_ials.fit(-np.inf, 0.9083779588159225)

ease_slimbpr = DifferentLossScoresHybridRecommender(URM_all_imp, EASE_R, SLIM_BPR_Cython)
ease_slimbpr.fit(np.inf, 0.9936801368250636)

slim_vae_itemuserknncf_ials_ease_slimbpr = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_itemuserknncf_ials, ease_slimbpr)
slim_vae_itemuserknncf_ials_ease_slimbpr.fit(1, 0.9760975034194325)

write_submission(recommender=slim_vae_itemuserknncf_ials_ease_slimbpr)