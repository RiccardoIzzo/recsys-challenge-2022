import scipy.sparse as sp
from skopt.space import Integer, Categorical, Real
from IPython.display import display
from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Content, \
    runHyperparameterSearch_Collaborative, runHyperparameterSearch_Hybrid
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

from Recommenders.Neural.MultVAERecommender import MultVAERecommender, MultVAERecommender_OptimizerMask
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender

from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from sklearn.preprocessing import normalize
from Utils.prep_sub import write_submission

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample


class ScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)
    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights


def linear_combination():
    ######################### DATA PREPARATION ###########################################

    # generate the dataframes for URM and ICMs
    dataReader = DataLoaderSplit(urm='interactionScoredOld.csv')
    URM, ICM_length, ICM_type = dataReader.get_csr_matrices()

    # concatenate the ICMs
    icm_mixed = sp.hstack([ICM_length, ICM_type])

    # split the URM df in training, validation and testing and turn it into a matrix
    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM, train_percentage=0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.8)

    urm_icm_stacked = sp.vstack([URM_train, icm_mixed.T])

    # stacked_URM = sps.csr_matrix(urm_icm_stacked)
    # stacked_ICM = sps.csr_matrix(stacked_URM.T)

    ############# LOAD OR FIT THE BASE MODELS ##############################################
    output_folder_path = "../trained_models/"

    from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    best_hyperparams_ItemKNNCF = {'topK': 785, 'shrink': 41, 'similarity': 'cosine', 'normalize': True}
    rec1 = ItemKNNCFRecommender(URM_train)
    rec1.fit(**best_hyperparams_ItemKNNCF)
    # rec1.load_model(folder_path="../trained_models/",
    #                 file_name=rec1.RECOMMENDER_NAME + "_cosine_best_model_last.zip")
    print("Recommender 1 is ready!")

    from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    rec2 = P3alphaRecommender(URM_train)
    rec2.fit(topK=633, alpha=0.0, normalize_similarity=True)
    # rec2.load_model(folder_path="../trained_models/", file_name=rec2.RECOMMENDER_NAME + "_best_model_last.zip")
    print("Recommender 2 is ready!")

    ############ TUNE THOSE HYPERPARAMETERS BABEH ##########################################

    # Step 1: Split the data and create the evaluator objects
    evaluator_validation = EvaluatorHoldout(URM_validation, [10])
    evaluator_test = EvaluatorHoldout(URM_test, [10])

    result_df, _ = evaluator_test.evaluateRecommender(rec1)
    print("{} FINAL MAP: {}".format(rec1.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    result_df, _ = evaluator_test.evaluateRecommender(rec2)
    print("{} FINAL MAP: {}".format(rec2.RECOMMENDER_NAME, result_df.loc[10]["MAP"]))

    # Step 2: Define hyperparameter set for the desired model, in this case ItemKNN
    hyperparameters_range_dictionary = {
        "alpha": Real(low=0.0, high=1.0, prior='uniform'),
    }

    # Step 3: Create SearchBayesianSkopt object, providing the desired recommender class and evaluator objects
    recommender_class = ScoresHybridRecommender

    hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                               evaluator_validation=evaluator_test,
                                               evaluator_test=evaluator_test)

    # Step 4: Provide the data needed to create an instance of the model, one trained only on URM_train, the other on URM_train_validation
    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, rec1, rec2],
        # For a CBF model simply put [URM_train, ICM_train]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )
    recommender_input_args_last_test = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, rec1, rec2],  # CBF: [URM, ICM], CF: [URM]
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    # Step 5: Create a result folder and select the number of cases (50 with 30% random is a good number)
    output_folder_path = "../trained_models/"
    n_cases = 10
    n_random_starts = int(n_cases * 0.3)
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    # # general?

    hyperparameterSearch.search(recommender_input_args,
                                recommender_input_args_last_test=recommender_input_args_last_test,
                                hyperparameter_search_space=hyperparameters_range_dictionary,
                                n_cases=n_cases,
                                n_random_starts=n_random_starts,
                                save_model="best",
                                resume_from_saved=False,
                                output_folder_path=output_folder_path,  # Where to save the results
                                output_file_name_root=recommender_class.RECOMMENDER_NAME,  # How to call the files
                                metric_to_optimize=metric_to_optimize,
                                cutoff_to_optimize=cutoff_to_optimize,
                                )


if __name__ == '__main__':
    linear_combination()
