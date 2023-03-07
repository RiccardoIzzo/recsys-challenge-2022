from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sps
import numpy as np
from Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
import matplotlib.pyplot as plt

data_loader = DataLoaderSplit(urm='newURM.csv')
urm_train_scored = sps.load_npz('../../data/dataset_split/URM_train_new.npz')
urm_train = sps.load_npz('../../data/dataset_split/URM_train_imp_new.npz')
urm_test = sps.load_npz('../../data/dataset_split/URM_test_new.npz')
evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], verbose=True)

rec = []
rec_names = []

# 0
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

best_hyperparams_ItemKNNCF_tversky = {'topK': 50, 'shrink': 0, 'similarity': 'tversky', 'normalize': True,
                                      'tversky_alpha': 0.2677659937917337, 'tversky_beta': 1.8732432514946602}
ItemKNNCF_tversky = ItemKNNCFRecommender(URM_train=urm_train)
ItemKNNCF_tversky.fit(**best_hyperparams_ItemKNNCF_tversky)
rec_names.append(ItemKNNCF_tversky.RECOMMENDER_NAME)
rec.append(ItemKNNCF_tversky)

# 1
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

best_hyperparams_ItemKNNCF_asymmetric = {'topK': 50, 'shrink': 77, 'similarity': 'asymmetric', 'normalize': True,
                                         'asymmetric_alpha': 0.3388995614775971, 'feature_weighting': 'TF-IDF'}
ItemKNNCF_asymmetric = ItemKNNCFRecommender(URM_train=urm_train_scored)
ItemKNNCF_asymmetric.fit(**best_hyperparams_ItemKNNCF_asymmetric)
rec_names.append(ItemKNNCF_asymmetric.RECOMMENDER_NAME)
rec.append(ItemKNNCF_asymmetric)

# 2
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
                                              'alpha': 0.0009354394779247897}
SLIMElasticNet = SLIMElasticNetRecommender(URM_train=urm_train)
SLIMElasticNet.load_model(folder_path="../../trained_models/",
                          file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_80_new.zip")
rec_names.append(SLIMElasticNet.RECOMMENDER_NAME)
rec.append(SLIMElasticNet)

# 3
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
EASE_R = EASE_R_Recommender(URM_train=urm_train)
EASE_R.load_model(folder_path="../../trained_models/", file_name=EASE_R.RECOMMENDER_NAME + "_best_80_new.zip")
rec_names.append(EASE_R.RECOMMENDER_NAME)
rec.append(EASE_R)

# 4
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log', 'alpha': 38.524701045378585,
                         'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
IALS = IALSRecommender(URM_train=urm_train_scored)
IALS.load_model(folder_path="../../trained_models/", file_name=IALS.RECOMMENDER_NAME + "_best_80_new")
rec_names.append(IALS.RECOMMENDER_NAME)
rec.append(IALS)

# 5
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
                            'normalize_similarity': True, 'implicit': True}
RP3beta = RP3betaRecommender(URM_train=urm_train)
RP3beta.fit(**best_hyperparams_RP3beta)
rec_names.append(RP3beta.RECOMMENDER_NAME)
rec.append(RP3beta)

# 6
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask

best_hyperparams_MultVAE = {'epochs': 300, 'learning_rate': 0.0001, 'l2_reg': 3.2983231409888774e-06,
                            'dropout': 0.6, 'total_anneal_steps': 100000, 'anneal_cap': 0.6, 'batch_size': 512,
                            'encoding_size': 243, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 1,
                            'max_parameters': 1750000000.0}
MultVAE = MultVAERecommender_OptimizerMask(URM_train=urm_train)
MultVAE.load_model(folder_path="../../trained_models/", file_name=MultVAE.RECOMMENDER_NAME + "_best_80_new")
rec_names.append(MultVAE.RECOMMENDER_NAME)
rec.append(MultVAE)

# 7
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender

best_hyperparams_UserKNNCF = {'topK': 759, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'TF-IDF'}
userknncf_cosine = UserKNNCFRecommender(URM_train=urm_train)
userknncf_cosine.fit(**best_hyperparams_UserKNNCF)
rec_names.append(userknncf_cosine.RECOMMENDER_NAME)
rec.append(userknncf_cosine)

# 8
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

best_hyperparams_SLIM_BPR_Cython = {'topK': 89, 'epochs': 300, 'symmetric': True, 'sgd_mode': 'adagrad',
                                    'lambda_i': 0.001, 'lambda_j': 9.773899284052082e-05,
                                    'learning_rate': 0.03500210788755942}
SLIM_BPR_Cython = SLIM_BPR_Cython(urm_train)
SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)
rec_names.append(SLIM_BPR_Cython.RECOMMENDER_NAME)
rec.append(SLIM_BPR_Cython)

# 9
slim_vae = DifferentLossScoresHybridRecommender(urm_train, SLIMElasticNet, MultVAE)
slim_vae.fit(1, 0.35)
rec_names.append("SLIM_VAE")
rec.append(slim_vae)

# 10
itemknncf_asymmetric_tversky = DifferentLossScoresHybridRecommender(urm_train, ItemKNNCF_asymmetric, ItemKNNCF_tversky)
itemknncf_asymmetric_tversky.fit(1, 0.65)
rec_names.append("ItemKNNCF_asymmetric_tversky")
rec.append(itemknncf_asymmetric_tversky)

# 11
slim_vae_ease = DifferentLossScoresHybridRecommender(urm_train, slim_vae, EASE_R)
slim_vae_ease.fit(1, 0.9)
rec_names.append("SLIM_VAE_EASE")
rec.append(slim_vae_ease)

# 12
slim_vae_ease_itemknncf = DifferentLossScoresHybridRecommender(urm_train, slim_vae_ease, itemknncf_asymmetric_tversky)
slim_vae_ease_itemknncf.fit(1, 0.9340960931960542)
rec_names.append("SLIM_VAE_EASE_ItemKNNCF")
rec.append(slim_vae_ease_itemknncf)

# 13
slim_vae_itemknncf = DifferentLossScoresHybridRecommender(urm_train, slim_vae, itemknncf_asymmetric_tversky)
slim_vae_itemknncf.fit(1, 0.95)
rec_names.append("SLIM_VAE_ItemKNNCF")
rec.append(slim_vae_itemknncf)

# 14
slim_vae_itemknncf_ials = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf, IALS)
slim_vae_itemknncf_ials.fit(1, 0.965)
rec_names.append("SLIM_VAE_ItemKNNCF_IALS")
rec.append(slim_vae_itemknncf_ials)

# 15
slim_vae_itemknncf_ials_rp3 = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf_ials, RP3beta)
slim_vae_itemknncf_ials_rp3.fit(1, 0.95)
rec_names.append("SLIM_VAE_ItemKNNCF_IALS_RP3")
rec.append(slim_vae_itemknncf_ials_rp3)

# 16
vae_userknncf = DifferentLossScoresHybridRecommender(urm_train, MultVAE, userknncf_cosine)
vae_userknncf.fit(1, 0.5)
rec_names.append("VAE_USERKNNCF")
rec.append(vae_userknncf)

slim_vae_itemknncf_ease = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf, EASE_R)
slim_vae_itemknncf_ease.fit(1, 0.925)
rec_names.append("SLIM_VAE_ItemKNNCF_EASE")
rec.append(slim_vae_itemknncf_ease)

slim_vae_itemknncf_rp3 = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf, RP3beta)
slim_vae_itemknncf_rp3.fit(1, 0.975)
rec_names.append("SLIM_VAE_ItemKNNCF_RP3")
rec.append(slim_vae_itemknncf_rp3)

slim_vae_itemknncf_bpr = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf, SLIM_BPR_Cython)
slim_vae_itemknncf_bpr.fit(1, 0.975)
rec_names.append("SLIM_VAE_ItemKNNCF_BPR")
rec.append(slim_vae_itemknncf_bpr)

slim_vae_itemknncf_ease_bpr = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf_ease, SLIM_BPR_Cython)
slim_vae_itemknncf_ease_bpr.fit(1, 0.995)
rec_names.append("SLIM_VAE_ItemKNNCF_EASE_BPR")
rec.append(slim_vae_itemknncf_ease_bpr)

slim_vae_itemknncf_ease_ials = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf_ease, IALS)
slim_vae_itemknncf_ease_ials.fit(1, 0.99)
rec_names.append("SLIM_VAE_ItemKNNCF_EASE_IALS")
rec.append(slim_vae_itemknncf_ease_ials)

slim_vae_itemknncf_ease_rp3 = DifferentLossScoresHybridRecommender(urm_train, slim_vae_itemknncf_ease, RP3beta)
slim_vae_itemknncf_ease_rp3.fit(1, 0.985)
rec_names.append("SLIM_VAE_ItemKNNCF_EASE_RP3")
rec.append(slim_vae_itemknncf_ease_rp3)

MAP_recommender_per_group = {}

collaborative_recommender_class = {
    rec_names[0]: rec[0],
    rec_names[1]: rec[1],
    rec_names[2]: rec[2],
    rec_names[3]: rec[3],
    rec_names[4]: rec[4],
    rec_names[5]: rec[5],
    rec_names[6]: rec[6],
    rec_names[7]: rec[7],
    rec_names[8]: rec[8],
    rec_names[9]: rec[9],
    rec_names[10]: rec[10],
    rec_names[11]: rec[11],
    rec_names[12]: rec[12],
    rec_names[13]: rec[13],
    rec_names[14]: rec[14],
    rec_names[15]: rec[15],
    rec_names[16]: rec[16],
}

n = [0, 20, 25, 30, 35, 40, 45, 50, 58, 75, 85, 100, 120, 140, 170, 210, 270, 350, 500, 1000]
m = [20, 25, 30, 35, 40, 45, 50, 58, 75, 85, 100, 120, 140, 170, 210, 270, 350, 500, 1000, 8000]

for i in range(len(n)):
    outside, inside = data_loader.get_user_outside_n_m_interaction(n[i], m[i])
    evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], ignore_users=outside)

    for label, recommender in collaborative_recommender_class.items():
        result_df, _ = evaluator_validation.evaluateRecommender(recommender)

        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[10]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[10]["MAP"]]
maps = []
for i in range(len(rec_names)):
    maps.append(MAP_recommender_per_group[rec_names[i]])

data_loader.plot_map_graph(rec_names, maps)
names = rec_names
maps = maps
_ = plt.figure(figsize=(16, 9))
for i in range(len(names)):
    results = maps[i]
    plt.scatter(x=np.arange(0, len(results)), y=results, label=names[i])
plt.ylabel('MAP')
plt.xlabel('User Group')
plt.legend()
plt.xlim(-0.5, 3.5)
plt.ylim(0.00 ,0.01)
plt.show()
'''
Ranges:
1) 0-20: 3874
2) 20-25: 3862
3) 25-30: 3286
4) 30-35: 2586
5) 35-40: 2091
6) 40-45: 1783
7) 45-50: 1470
8) 50-58: 2090
9) 58-75: 3293
10) 75-85: 1564
11) 85-100: 1873
12) 100-120: 1923
13) 120-140: 1531
14) 140-170: 1679
15) 170-210: 1661
16) 210-270: 1767
17) 270-350: 1513
18) 350-500: 1579
19) 500-1000: 1560
20) 1000-8000: 649
'''
