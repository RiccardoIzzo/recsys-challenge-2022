from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, \
    runHyperparameterSearch_Hybrid
from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.DataIO import DataIO
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.DataIO import DataIO
from skopt.space import Real, Integer, Categorical
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender, MultVAERecommender_OptimizerMask
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender, \
    SLIMElasticNetRecommender


def main():
    dataReader = DataLoaderSplit(urm='newURM.csv')
    URM_all, ICM_length, ICM_type = dataReader.get_csr_matrices()
    URM_all = dataReader.get_implicit_single_mixed_csr()
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # import scipy.sparse as sp
    # icm_mixed = sp.hstack([ICM_length, ICM_type])
    # urm_icm_stacked = sp.vstack([URM_train, icm_mixed.T])
    # urm_stacked_mix_train = sp.vstack([URM_train, icm_mixed.T])
    # urm_stacked_mix_train_validation = sp.vstack([URM_train_validation, icm_mixed.T])

    ### CHOOSE RECOMMENDER HERE ###
    recommender_class = MultVAERecommender_OptimizerMask

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test)

    n_cases = 50
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    similarity_type_list=['cosine', 'jaccard', "asymmetric", "dice", "tversky", "tanimoto"],

    hyperparameters_range_dictionary = {
        "epochs": Categorical([500]),
        "learning_rate": Real(low=1e-6, high=1e-2, prior="log-uniform"),
        "l2_reg": Real(low=1e-6, high=1e-2, prior="log-uniform"),
        "dropout": Real(low=0., high=0.8, prior="uniform"),
        "total_anneal_steps": Integer(100000, 600000),
        "anneal_cap": Real(low=0., high=0.6, prior="uniform"),
        "batch_size": Categorical([128, 256, 512, 1024]),

        #"encoding_size": Integer(1, min(512, n_items - 1)),
        "next_layer_size_multiplier": Integer(2, 10),
        "max_n_hidden_layers": Integer(1, 4),

        # Constrain the model to a maximum number of parameters so that its size does not exceed 7 GB
        # Estimate size by considering each parameter uses float32
        "max_parameters": Categorical([7 * 1e9 * 8 / 32]),
    }

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],  # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )

    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train_validation],
        # For a CBF model simply put [URM_train_validation, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={},
        EARLYSTOPPING_KEYWORD_ARGS={},
    )
    #
    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="best",
                                output_folder_path='../trained_models/',  # Where to save the results
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                resume_from_saved=True,
                                )

    runHyperparameterSearch_Collaborative(recommender_class,
                                          URM_train=URM_train,
                                          URM_train_last_test=URM_train_validation,
                                          metric_to_optimize=metric_to_optimize,
                                          cutoff_to_optimize=cutoff_to_optimize,
                                          #evaluator_validation_earlystopping=evaluator_validation,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test,
                                          output_folder_path='../trained_models/',
                                          parallelizeKNN=False,
                                          allow_weighting=True,
                                          resume_from_saved=True,
                                          save_model="best",
                                          similarity_type_list=['cosine'],
                                          n_cases=n_cases,
                                          n_random_starts=n_random_starts)

    # CBF and so on...
    # runHyperparameterSearch_Content(recommender_class,
    #                                 URM_train=urm_train,
    #                                 ICM_object=icm_mixed_train,
    #                                 ICM_name="mixed",
    #                                 URM_train_last_test=urm_train_validation,
    #                                 metric_to_optimize=metric_to_optimize,
    #                                 cutoff_to_optimize=cutoff_to_optimize,
    #                                 evaluator_validation=evaluator_validation,
    #                                 evaluator_test=evaluator_test,
    #                                 output_folder_path='../trained_models/',
    #                                 resume_from_saved=True,
    #                                 save_model="best",
    #                                 n_cases=n_cases,
    #                                 n_random_starts=n_random_starts,
    #                                 evaluate_on_test="best",
    #                                 max_total_time=None,
    #                                 parallelizeKNN=True,
    #                                 allow_weighting=True, allow_bias_ICM=False,
    #                                 similarity_type_list=None
    #                                 )

    # runHyperparameterSearch_Hybrid(recommender_class,
    #                                URM_train=urm_train,
    #                                ICM_object=icm_mixed_train,
    #                                ICM_name="mixed",
    #                                URM_train_last_test=urm_train_validation,
    #                                n_cases=n_cases,
    #                                n_random_starts=n_random_starts,
    #                                resume_from_saved=True,
    #                                save_model="best",
    #                                evaluator_validation_earlystopping=evaluator_validation,
    #                                evaluator_validation=evaluator_validation,
    #                                evaluator_test=evaluator_test,
    #                                metric_to_optimize=metric_to_optimize,
    #                                cutoff_to_optimize=cutoff_to_optimize,
    #                                output_folder_path='../trained_models/',
    #                                parallelizeKNN=False,
    #                                allow_weighting=True)

    ################### VISUALIZE RESULTS FOR HYPERPARAMETER SEARCH ########################

    # data_loader = DataIO(folder_path='../trained_models')
    # search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")
    # search_metadata.keys()
    #
    # hyperparameters_df = search_metadata["hyperparameters_df"]
    # print(hyperparameters_df)
    #
    # result_on_validation_df = search_metadata["result_on_validation_df"]
    # print(result_on_validation_df)
    #
    # result_best_on_test = search_metadata["result_on_last"]
    # print(result_best_on_test)
    #
    # best_hyperparameters = search_metadata["hyperparameters_best"]
    # print(best_hyperparameters)
    #
    # print("FINAL MAP: {}".format(result_best_on_test.loc[10]["MAP"]))


if __name__ == "__main__":
    main()
