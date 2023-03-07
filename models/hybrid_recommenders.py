from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from numpy import linalg as LA
import numpy as np
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from skopt.space import Categorical, Real


def main():
    # FOR GRID SEARCH
    dataReader = DataLoaderSplit(urm='newURM.csv')
    URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
    # dataReader = DataLoaderSplit(urm='newURM.csv')
    # URM_all = dataReader.get_implicit_single_mixed_csr()
    # URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)
    URM_train = sps.load_npz('../data/dataset_split/URM_train_new.npz')
    URM_train_imp = sps.load_npz('../data/dataset_split/URM_train_imp_new.npz')
    URM_test = sps.load_npz('../data/dataset_split/URM_test_new.npz')
    evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10], verbose=True)

    # FOR BAYESIAN OPTIMIZATION WITH SKOPT
    # dataReader = DataLoaderSplit(urm='LastURM.csv')
    # URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
    # URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)
    # evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    # evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    class DifferentLossScoresHybridRecommender(BaseRecommender):
        """ ScoresHybridRecommender
        Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
        algorithms trained on different loss functions.

        """

        RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"

        def __init__(self, URM_train, recommender_1, recommender_2):
            super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

            self.URM_train = sps.csr_matrix(URM_train)
            self.recommender_1 = recommender_1
            self.recommender_2 = recommender_2

        def fit(self, norm, alpha=0.5):

            self.alpha = alpha
            self.norm = norm

        def _compute_item_score(self, user_id_array, items_to_compute):

            item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)

            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

            item_weights = []
            from tqdm import trange
            for item in trange(len(item_weights_1[0])):
                if item_weights_1[0][item] == 0:
                    item_weights.append([item_weights_2[0][item] / norm_item_weights_2])
                elif item_weights_2[0][item] == 0:
                    item_weights.append(item_weights_1[0][item] / norm_item_weights_1)
                else:
                    item_weights.append(item_weights_1[0][item] / norm_item_weights_1 * self.alpha + item_weights_2[0][item] / norm_item_weights_2 * (
                            1 - self.alpha))

            item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (
                    1 - self.alpha)

            return item_weights

    ### INSERT YOUR RECOMMENDERS WITH THE BEST HYPERPARAMS HERE ###

    # ItemKNNCFRecommender
    # asymmetric
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    best_hyperparams_ItemKNNCF_asymmetric = {'topK': 50, 'shrink': 77, 'similarity': 'asymmetric', 'normalize': True,
                                             'asymmetric_alpha': 0.3388995614775971, 'feature_weighting': 'TF-IDF'}
    ItemKNNCF_asymmetric = ItemKNNCFRecommender(URM_train=URM_train)
    ItemKNNCF_asymmetric.fit(**best_hyperparams_ItemKNNCF_asymmetric)
    result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF_asymmetric)
    print("{} FINAL MAP: {}".format(ItemKNNCF_asymmetric.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # tversky
    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    best_hyperparams_ItemKNNCF_tversky = {'topK': 50, 'shrink': 0, 'similarity': 'tversky', 'normalize': True,
                                          'tversky_alpha': 0.2677659937917337, 'tversky_beta': 1.8732432514946602}
    ItemKNNCF_tversky = ItemKNNCFRecommender(URM_train=URM_train)
    ItemKNNCF_tversky.fit(**best_hyperparams_ItemKNNCF_tversky)
    result_df, _ = evaluator_validation.evaluateRecommender(ItemKNNCF_tversky)
    print("{} FINAL MAP: {}".format(ItemKNNCF_tversky.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # SLIMElasticNetRecommender
    from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
    # from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
    best_hyperparams_SLIMElasticNetRecommender = {'topK': 2371, 'l1_ratio': 9.789016848114697e-07,
                                                  'alpha': 0.0009354394779247897}
    SLIMElasticNet = SLIMElasticNetRecommender(URM_train=URM_train_imp)
    # SLIMElasticNet = MultiThreadSLIM_SLIMElasticNetRecommender(URM_train=URM_train)
    #SLIMElasticNet.load_model(folder_path="../trained_models/", file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_80")
    SLIMElasticNet.fit(**best_hyperparams_SLIMElasticNetRecommender)
    result_df, _ = evaluator_validation.evaluateRecommender(SLIMElasticNet)
    print("{} FINAL MAP: {}".format(SLIMElasticNet.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
    #SLIMElasticNet.save_model("../trained_models/", file_name=SLIMElasticNet.RECOMMENDER_NAME + "_best_80_new")

    # EASE_R_Recommender
    from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
    best_hyperparams_EASE_R = {'topK': 2584, 'normalize_matrix': False, 'l2_norm': 163.5003203521832}
    EASE_R = EASE_R_Recommender(URM_train=URM_train)
    EASE_R.load_model(folder_path="../trained_models/", file_name=EASE_R.RECOMMENDER_NAME + "_best_80")
    # EASE_R.fit(**best_hyperparams_EASE_R)
    result_df, _ = evaluator_validation.evaluateRecommender(EASE_R)
    print("{} FINAL MAP: {}".format(EASE_R.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # IALS
    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    best_hyperparams_IALS = {'num_factors': 193, 'epochs': 80, 'confidence_scaling': 'log', 'alpha': 38.524701045378585,
                             'epsilon': 0.11161267696066449, 'reg': 0.00016885775864831462}
    IALS = IALSRecommender(URM_train=URM_train)
    IALS.load_model(folder_path="../trained_models/", file_name=IALS.RECOMMENDER_NAME + "_best_80")
    IALS.fit(**best_hyperparams_IALS)
    result_df, _ = evaluator_validation.evaluateRecommender(IALS)
    print("{} FINAL MAP: {}".format(IALS.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
    IALS.save_model("../trained_models/", file_name=IALS.RECOMMENDER_NAME + "_best_80_new")

    # RP3beta
    from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    best_hyperparams_RP3beta = {'topK': 181, 'alpha': 0.8283771829787758, 'beta': 0.46337458582020374,
                                'normalize_similarity': True, 'implicit': True}
    RP3beta = RP3betaRecommender(URM_train=URM_train)
    RP3beta.load_model(folder_path="../trained_models/", file_name=RP3beta.RECOMMENDER_NAME + "_best_80")
    RP3beta.fit(**best_hyperparams_RP3beta)
    result_df, _ = evaluator_validation.evaluateRecommender(RP3beta)
    print("{} FINAL MAP: {}".format(RP3beta.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # MultVAE
    from Recommenders.Neural.MultVAERecommender import MultVAERecommender_OptimizerMask
    best_hyperparams_MultVAE = {'epochs': 300, 'learning_rate': 0.0001, 'l2_reg': 3.2983231409888774e-06,
                                'dropout': 0.6, 'total_anneal_steps': 100000, 'anneal_cap': 0.6, 'batch_size': 512,
                                'encoding_size': 243, 'next_layer_size_multiplier': 2, 'max_n_hidden_layers': 1,
                                'max_parameters': 1750000000.0}
    MultVAE = MultVAERecommender_OptimizerMask(URM_train=URM_train_imp)
    MultVAE.load_model(folder_path="../trained_models/", file_name=MultVAE.RECOMMENDER_NAME + "_best_80")
    MultVAE.fit(**best_hyperparams_MultVAE)
    result_df, _ = evaluator_validation.evaluateRecommender(MultVAE)
    print("{} FINAL MAP: {}".format(MultVAE.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))
    MultVAE.save_model("../trained_models/", file_name=MultVAE.RECOMMENDER_NAME + "_best_80_new")

    # SLIM_BPR_Cython
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    best_hyperparams_SLIM_BPR_Cython = {'topK': 89, 'epochs': 300, 'symmetric': True, 'sgd_mode': 'adagrad',
                                        'lambda_i': 0.001, 'lambda_j': 9.773899284052082e-05,
                                        'learning_rate': 0.03500210788755942}
    SLIM_BPR_Cython = SLIM_BPR_Cython(URM_train)
    SLIM_BPR_Cython.fit(**best_hyperparams_SLIM_BPR_Cython)
    result_df, _ = evaluator_validation.evaluateRecommender(SLIM_BPR_Cython)
    print("{} FINAL MAP: {}".format(SLIM_BPR_Cython.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # UserKNNCFRecommender
    from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
    best_hyperparams_UserKNNCF = {'topK': 759, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                                  'feature_weighting': 'TF-IDF'}
    userknncf_cosine = UserKNNCFRecommender(URM_train=URM_train)
    userknncf_cosine.fit(**best_hyperparams_UserKNNCF)

    # P3alpha
    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    best_hyperparams_P3alpha = {'topK': 213, 'alpha': 0.7806594643806142, 'normalize_similarity': True,
                                'implicit': True}
    P3alpha = P3alphaRecommender(URM_train=URM_train)
    P3alpha.fit(**best_hyperparams_P3alpha)
    result_df, _ = evaluator_validation.evaluateRecommender(P3alpha)
    print("{} FINAL MAP: {}".format(P3alpha.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    class ScoresHybridRecommender_3(BaseRecommender):
        RECOMMENDER_NAME = "ScoresHybridRecommender_3"

        def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
            super(ScoresHybridRecommender_3, self).__init__(URM_train)

            self.alpha = None
            self.beta = None
            self.gamma = None
            self.norm = None

            self.URM_train = sps.csr_matrix(URM_train)
            self.recommender_1 = recommender_1
            self.recommender_2 = recommender_2
            self.recommender_3 = recommender_3

        def fit(self, norm, alpha=0.5, beta=0.5, gamma=0.5):
            self.norm = norm
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        def _compute_item_score(self, user_id_array, items_to_compute):
            # In a simple extension this could be a loop over a list of pretrained recommender objects
            item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
            item_weights_3 = self.recommender_3._compute_item_score(user_id_array)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
            norm_item_weights_3 = LA.norm(item_weights_3, self.norm)

            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))

            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))

            if norm_item_weights_3 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 3 is zero. Avoiding division by zero".format(self.norm))

            item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha + \
                           (item_weights_2 / norm_item_weights_2) * self.beta + \
                           (item_weights_3 / norm_item_weights_3) * self.gamma

            return item_weights

        def save_model(self, folder_path, file_name=None):
            return

        def load_model(self, folder_path, file_name=None):
            return

    class ScoresHybridRecommender_7(BaseRecommender):
        RECOMMENDER_NAME = "ScoresHybridRecommender_7"

        def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, recommender_4, recommender_5,
                     recommender_6, recommender_7):
            super(ScoresHybridRecommender_7, self).__init__(URM_train)

            self.alpha = None
            self.beta = None
            self.gamma = None
            self.theta = None
            self.omega = None
            self.phi = None
            self.rho = None
            self.norm = None

            self.URM_train = sps.csr_matrix(URM_train)
            self.recommender_1 = recommender_1
            self.recommender_2 = recommender_2
            self.recommender_3 = recommender_3
            self.recommender_4 = recommender_4
            self.recommender_5 = recommender_5
            self.recommender_6 = recommender_6
            self.recommender_7 = recommender_7

        def fit(self, norm, alpha=0.5, beta=0.5, gamma=0.5, theta=0.5, omega=0.5, phi=0.5, rho=0.5):
            self.norm = norm
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.theta = theta
            self.omega = omega
            self.phi = phi
            self.rho = rho

        def _compute_item_score(self, user_id_array, items_to_compute):
            # In a simple extension this could be a loop over a list of pretrained recommender objects
            item_weights_1 = self.recommender_1._compute_item_score(user_id_array, items_to_compute)
            item_weights_2 = self.recommender_2._compute_item_score(user_id_array, items_to_compute)
            item_weights_3 = self.recommender_3._compute_item_score(user_id_array, items_to_compute)
            item_weights_4 = self.recommender_4._compute_item_score(user_id_array, items_to_compute)
            item_weights_5 = self.recommender_5._compute_item_score(user_id_array, items_to_compute)
            item_weights_6 = self.recommender_6._compute_item_score(user_id_array, items_to_compute)
            item_weights_7 = self.recommender_7._compute_item_score(user_id_array, items_to_compute)

            norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
            norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
            norm_item_weights_3 = LA.norm(item_weights_3, self.norm)
            norm_item_weights_4 = LA.norm(item_weights_4, self.norm)
            norm_item_weights_5 = LA.norm(item_weights_5, self.norm)
            norm_item_weights_6 = LA.norm(item_weights_6, self.norm)
            norm_item_weights_7 = LA.norm(item_weights_7, self.norm)

            if norm_item_weights_1 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_2 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_3 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 3 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_4 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 4 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_5 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 5 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_6 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 6 is zero. Avoiding division by zero".format(
                        self.norm))

            if norm_item_weights_7 == 0:
                raise ValueError(
                    "Norm {} of item weights for recommender 7 is zero. Avoiding division by zero".format(
                        self.norm))

            item_weights = (item_weights_1 / norm_item_weights_1) * self.alpha + \
                           (item_weights_2 / norm_item_weights_2) * self.beta + \
                           (item_weights_3 / norm_item_weights_3) * self.gamma + \
                           (item_weights_4 / norm_item_weights_4) * self.theta + \
                           (item_weights_5 / norm_item_weights_5) * self.omega + \
                           (item_weights_6 / norm_item_weights_6) * self.phi + \
                           (item_weights_7 / norm_item_weights_7) * self.rho

            return item_weights

        def save_model(self, folder_path, file_name=None):
            return

        def load_model(self, folder_path, file_name=None):
            return

    slim_vae = DifferentLossScoresHybridRecommender(URM_train, SLIMElasticNet, MultVAE)
    slim_vae.fit(1, 0.35189378875691635)

    itemknncf_asymmetric_tversky = DifferentLossScoresHybridRecommender(URM_train, ItemKNNCF_asymmetric, ItemKNNCF_tversky)
    itemknncf_asymmetric_tversky.fit(1, 0.6384198928672995)

    slim_vae_itemknncf = DifferentLossScoresHybridRecommender(URM_train, slim_vae, itemknncf_asymmetric_tversky)
    slim_vae_itemknncf.fit(1, 0.9418377596234007)

    slim_vae_itemknncf_ials = DifferentLossScoresHybridRecommender(URM_train, slim_vae_itemknncf, IALS)
    slim_vae_itemknncf_ials.fit(np.inf, 0.9381147385789094)

    slim_vae_itemknncf_ials_rp3 = DifferentLossScoresHybridRecommender(URM_train, slim_vae_itemknncf_ials, RP3beta)
    slim_vae_itemknncf_ials_rp3.fit(1, 0.9559876377330556)

    slim_vae_itemknncf_ials_rp3_ease = DifferentLossScoresHybridRecommender(URM_train, slim_vae_itemknncf_ials_rp3, EASE_R)
    slim_vae_itemknncf_ials_rp3_ease.fit(1, 0.9936156064497043)

    # GRID SEARCH (norm, alpha)
    recommender_object = DifferentLossScoresHybridRecommender(URM_train, SLIMElasticNet, MultVAE)
    for norm in [1, 2, np.inf, -np.inf]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recommender_object.fit(norm, alpha)
            result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
            print("Norm: {}, Alpha: {}, Result: {}".format(norm, alpha, result_df.loc[10]["MAP"]))

    # SKOPT SEARCH (norm, alpha)
    hyperparameters_range_dictionary = {
        "norm": Categorical([1, 2, np.inf, -np.inf]),
        "alpha": Real(low=0.0, high=1.0, prior='uniform'),
    }

    recommender_class = DifferentLossScoresHybridRecommender
    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=None)

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, SLIMElasticNet, MultVAE],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )
    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, SLIMElasticNet, MultVAE],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    output_folder_path = "../trained_models/"
    n_cases = 100
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="best",
                                resume_from_saved=False,
                                output_folder_path=output_folder_path,
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )


# necessary for multiprocessing (MultiThreadSLIMElasticNet)
if __name__ == "__main__":
    main()