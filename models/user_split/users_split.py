from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Hybrid.DifferentLossScoresHybridRecommender import DifferentLossScoresHybridRecommender
import numpy as np


def main():
    n = [0, 20, 50, 100, 500, 1000, 2000]
    m = [20, 50, 100, 500, 1000, 2000, 7799]
    data_loader = DataLoaderSplit(urm='newURM.csv')

    for i in range(len(n)):
        urm1, urm2, list = data_loader.get_matrix_between_n_m_interaction(n[i], m[i])
        urmscore, urm_test = split_train_in_two_percentage_global_sample(urm1, 0.8)
        evaluator_validation = EvaluatorHoldout(urm_test, cutoff_list=[10], verbose=True)

        urmimp = urmscore.copy()
        for k in range(len(urmimp.data)):
            urmimp.data[k] = 1

        from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
        best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log',
                                 'alpha': 38.524701045378585,
                                 'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
        IALS = IALSRecommender(URM_train=urmscore)
        IALS.fit(**best_hyperparams_IALS)

        from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
        best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
                                                      'alpha': 0.0009354394779247897}
        SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=urmimp)
        SLIMElasticNet.fit(**best_hyperparams_SLIMElasticNetRecommender)

        from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
        best_hyperparams_MultVAE = {'epochs': 300, 'learning_rate': 0.0001, 'l2_reg': 3.2983231409888774e-06,
                                    'dropout': 0.6, 'total_anneal_steps': 100000, 'anneal_cap': 0.6, 'batch_size': 512,
                                    'encoding_size': 243, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 1,
                                    'max_parameters': 1750000000.0}
        MultVAE = MultVAERecommender_OptimizerMask(URM_train=urmimp)
        MultVAE.fit(**best_hyperparams_MultVAE)

        from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
        best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
        EASE_R = EASE_R_Recommender(URM_train=urmimp)
        EASE_R.fit(**best_hyperparams_EASE_R)

        from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
        best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
                                    'normalize_similarity': True, 'implicit': True}
        RP3beta = RP3betaRecommender(URM_train=urmimp)
        RP3beta.fit(**best_hyperparams_RP3beta)

        from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
        best_hyperparams_ItemKNNCF_asymmetric = {'topK': 50, 'shrink': 77, 'similarity': 'asymmetric',
                                                 'normalize': True,
                                                 'asymmetric_alpha': 0.3388995614775971, 'feature_weighting': 'TF-IDF'}
        ItemKNNCF_asymmetric = ItemKNNCFRecommender(URM_train=urmscore)
        ItemKNNCF_asymmetric.fit(**best_hyperparams_ItemKNNCF_asymmetric)

        from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
        best_hyperparams_ItemKNNCF_tversky = {'topK': 50, 'shrink': 0, 'similarity': 'tversky', 'normalize': True,
                                              'tversky_alpha': 0.2677659937917337, 'tversky_beta': 1.8732432514946602}
        ItemKNNCF_tversky = ItemKNNCFRecommender(URM_train=urmimp)
        ItemKNNCF_tversky.fit(**best_hyperparams_ItemKNNCF_tversky)

        itemknncf_asymmetric_tversky = DifferentLossScoresHybridRecommender(urmimp, ItemKNNCF_asymmetric,
                                                                            ItemKNNCF_tversky)
        itemknncf_asymmetric_tversky.fit(1, 0.6384198928672995)

        ### BEST HYBRID
        slim_vae = DifferentLossScoresHybridRecommender(urmimp, SLIMElasticNet, MultVAE)
        slim_vae.fit(1, 0.35)

        slim_vae_itemknncf = DifferentLossScoresHybridRecommender(urmimp, slim_vae, itemknncf_asymmetric_tversky)
        slim_vae_itemknncf.fit(1, 0.95)

        slim_vae_itemknncf_ials = DifferentLossScoresHybridRecommender(urmimp, slim_vae_itemknncf, IALS)
        slim_vae_itemknncf_ials.fit(1, 0.965)

        slim_vae_itemknncf_ials_rp3 = DifferentLossScoresHybridRecommender(urmimp, slim_vae_itemknncf_ials, RP3beta)
        slim_vae_itemknncf_ials_rp3.fit(1, 0.95)

        slim_vae_itemknncf_ials_rp3_ease = DifferentLossScoresHybridRecommender(urmimp, slim_vae_itemknncf_ials_rp3,
                                                                                EASE_R)
        slim_vae_itemknncf_ials_rp3_ease.fit(1, 0.9936156064497043)

        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_itemknncf)
        print("slim_vae_itemknncf - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                            result_dict.loc[10]["MAP"]))

        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_itemknncf_ials)
        print("slim_vae_itemknncf_ials - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                 result_dict.loc[10]["MAP"]))
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_itemknncf_ials_rp3)
        print("slim_vae_itemknncf_ials_rp3 - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                     result_dict.loc[10]["MAP"]))
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_itemknncf_ials_rp3_ease)
        print("slim_vae_itemknncf_ials_rp3_ease - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                          result_dict.loc[10]["MAP"]))

        slim_itemknncf = DifferentLossScoresHybridRecommender(urmimp, SLIMElasticNet, itemknncf_asymmetric_tversky)
        slim_itemknncf.fit(-np.inf, 0.9319153610052522)
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_itemknncf)
        print("slim_itemknncf - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i], result_dict.loc[10]["MAP"]))

        slim_itemknncf_rp3 = DifferentLossScoresHybridRecommender(urmimp, slim_itemknncf, RP3beta)
        slim_itemknncf_rp3.fit(2, 0.8559468469346905)
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_itemknncf_rp3)
        print("slim_itemknncf_rp3 - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i], result_dict.loc[10]["MAP"]))

        slim_vae_ease = DifferentLossScoresHybridRecommender(urmimp, slim_vae, EASE_R)
        slim_vae_ease.fit(1, 0.9)

        slim_vae_ease_itemknncf = DifferentLossScoresHybridRecommender(urmimp, slim_vae_ease,
                                                                       itemknncf_asymmetric_tversky)
        slim_vae_ease_itemknncf.fit(1, 0.9340960931960542)
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_ease_itemknncf)
        print("slim_vae_ease_itemknncf - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                 result_dict.loc[10]["MAP"]))

        slim_vae_ease_itemknncf_ials = DifferentLossScoresHybridRecommender(urmimp, slim_vae_ease_itemknncf, IALS)
        slim_vae_ease_itemknncf_ials.fit(-np.inf, 0.9053260040059369)
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_ease_itemknncf_ials)
        print("slim_vae_ease_itemknncf_ials - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                      result_dict.loc[10]["MAP"]))

        slim_vae_ease_itemknncf_rp3 = DifferentLossScoresHybridRecommender(urmimp, slim_vae_ease_itemknncf, RP3beta)
        slim_vae_ease_itemknncf_rp3.fit(1, 0.93)
        result_dict, _ = evaluator_validation.evaluateRecommender(slim_vae_ease_itemknncf_rp3)
        print("slim_vae_ease_itemknncf_rp3 - i: {}. n: {}, m: {}, Result: {}".format(i, n[i], m[i],
                                                                                     result_dict.loc[10]["MAP"]))


if __name__ == "__main__":
    main()
