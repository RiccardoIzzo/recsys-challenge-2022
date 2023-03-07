from Utils.DataLoaderSplit import DataLoaderSplit
from Utils.prep_sub import *
from Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender

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

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
                                              'alpha': 0.0009354394779247897}
SLIMElasticNet = SLIMElasticNetRecommender(URM_train=URM_all_imp)
# SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=URM_train)
SLIMElasticNet.load_model(folder_path="../../trained_models/", file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_100")

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
EASE_R = EASE_R_Recommender(URM_train=URM_all_imp)
EASE_R.load_model(folder_path="../../trained_models/", file_name=EASE_R.RECOMMENDER_NAME + "_best_100")

from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask

best_hyperparams_MultVAE = {'epochs': 300, 'learning_rate': 0.0001, 'l2_reg': 3.2983231409888774e-06,
                            'dropout': 0.6, 'total_anneal_steps': 100000, 'anneal_cap': 0.6, 'batch_size': 512,
                            'encoding_size': 243, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 1,
                            'max_parameters': 1750000000.0}
MultVAE = MultVAERecommender_OptimizerMask(URM_train=URM_all_imp)
MultVAE.load_model(folder_path="../../trained_models/", file_name=MultVAE.RECOMMENDER_NAME + "_best_100")

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
                            'normalize_similarity': True, 'implicit': True}
RP3beta = RP3betaRecommender(URM_train=URM_all_imp)
RP3beta.fit(**best_hyperparams_RP3beta)

from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log', 'alpha': 38.524701045378585,
                         'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
IALS = IALSRecommender(URM_train=URM_all)
IALS.load_model(folder_path="../../trained_models/", file_name=IALS.RECOMMENDER_NAME + "_best_100_new")

from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

best_hyperparams_UserKNNCF = {'topK': 759, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}
userknncf_cosine = UserKNNCFRecommender(URM_train=URM_all_imp)
userknncf_cosine.fit(**best_hyperparams_UserKNNCF)

from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
best_hyperparams_SLIM_BPR_Cython = {'topK': 89, 'epochs': 300, 'symmetric': True, 'sgd_mode': 'adagrad',
                                    'lambda_i': 0.001, 'lambda_j': 9.773899284052082e-05,
                                    'learning_rate': 0.03500210788755942}
SLIM_BPR_Cython = SLIM_BPR_Cython(URM_all_imp)
SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)

ease_vae = DifferentLossScoresHybridRecommender(URM_all_imp, EASE_R, MultVAE)
ease_vae.fit(2, 0.17696115009843233)

itemknncf_asymmetric_tversky = DifferentLossScoresHybridRecommender(URM_all_imp, ItemKNNCF_asymmetric, ItemKNNCF_tversky)
itemknncf_asymmetric_tversky.fit(1, 0.65)

slim_vae = DifferentLossScoresHybridRecommender(URM_all_imp, SLIMElasticNet, MultVAE)
slim_vae.fit(1, 0.35)

slim_itemknncf = DifferentLossScoresHybridRecommender(URM_all_imp, SLIMElasticNet, itemknncf_asymmetric_tversky)
slim_itemknncf.fit(1, 0.75)

slim_vae_ease = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae, EASE_R)
slim_vae_ease.fit(1, 0.9)

slim_vae_ease_itemknncf = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_ease, itemknncf_asymmetric_tversky)
slim_vae_ease_itemknncf.fit(1, 0.9244859600516322)

#tune better
slim_vae_ease_itemknncf_rp3 = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_ease_itemknncf, RP3beta)
slim_vae_ease_itemknncf_rp3.fit(1, 0.93)

#tune better
slim_vae_itemknncf = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae, itemknncf_asymmetric_tversky)
slim_vae_itemknncf.fit(1, 0.95)

slim_vae_itemknncf_ials = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_itemknncf, IALS)
slim_vae_itemknncf_ials.fit(1, 0.965)

slim_vae_itemknncf_ials_rp3 = DifferentLossScoresHybridRecommender(URM_all_imp, slim_vae_itemknncf_ials, RP3beta)
slim_vae_itemknncf_ials_rp3.fit(1, 0.95)

vae_userknncf = DifferentLossScoresHybridRecommender(URM_all_imp, MultVAE, userknncf_cosine)
vae_userknncf.fit(1, 0.5)


urm1, urm2, list1 = dataReader.get_matrix_between_n_m_interaction(0, 25)
urm1, urm2, list2 = dataReader.get_matrix_between_n_m_interaction(25, 30)
urm1, urm2, list3 = dataReader.get_matrix_between_n_m_interaction(30, 58)
urm1, urm2, list4 = dataReader.get_matrix_between_n_m_interaction(58, 120)
urm1, urm2, list5 = dataReader.get_matrix_between_n_m_interaction(120, 350)
urm1, urm2, list6 = dataReader.get_matrix_between_n_m_interaction(350, 1000)
urm1, urm2, list7 = dataReader.get_matrix_between_n_m_interaction(1000, 8000)


write_submission_split_opt(MultVAE, slim_vae_ease_itemknncf, slim_vae_ease_itemknncf_rp3, slim_vae_ease_itemknncf, slim_vae_ease, slim_vae_itemknncf, slim_vae_ease_itemknncf_rp3, list1, list2, list3, list4, list5, list6, list7)