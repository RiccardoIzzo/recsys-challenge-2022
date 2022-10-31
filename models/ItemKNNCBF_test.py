from Utils.DataLoaderSplit import DataLoaderSplit
from Evaluation.Evaluator import EvaluatorHoldout
from Utils.prep_sub import write_submission

# Hyperparameter
test_split = 0.2
validation_split = 0.1
shrink = 100000
slice_size = 100

# URM_all_dataframe, ICM_dataframe2, ICM_dataframe = read_data()
data_loader = DataLoaderSplit()
iai = data_loader.get_iai()
train, test, val, n_episode_list, ICM_dataframe = data_loader.get_all_csr_matrices()

evaluator_validation = EvaluatorHoldout(val, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(train, cutoff_list=[10])

# ##############################################	recommender part	 ############################################# #

from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from bayes_opt import BayesianOptimization

recommender = ItemKNNCBFRecommender(URM_train=train, ICM_train=ICM_dataframe, verbose=False)

MAP_per_k_grid = []
i = 0

tuning_params = {
    "topK": (10, 500),
    "shrink": (0, 200)
}

hyperparameters = {'target': 0.0}

for similarity in ['cosine', 'pearson', 'jaccard', 'tanimoto', 'adjusted', 'euclidean']:
    for feature_weighting in ["BM25", "TF-IDF", "none"]:

        def BO_func(topK, shrink):
            recommender.fit(topK=int(topK), shrink=shrink, similarity=similarity, feature_weighting=feature_weighting)
            result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
            print(result_dict.loc[10])

            return result_dict.loc[10]["MAP"]


        optimizer = BayesianOptimization(
            f=BO_func,
            pbounds=tuning_params,
            verbose=5,
            random_state=5,
        )

        from bayes_opt.logger import JSONLogger
        from bayes_opt.event import Events

        logger = JSONLogger(
            path="logs/" + recommender.RECOMMENDER_NAME + '_' + similarity + '_' + feature_weighting + "_logs.json"
        )
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        optimizer.maximize(
            init_points=10,
            n_iter=10,
        )

        if optimizer.max['target'] > hyperparameters['target']:
            hyperparameters = optimizer.max
            hyperparameters['params']['similarity'] = similarity
            hyperparameters['params']['feature_weighting'] = feature_weighting
            print(hyperparameters)

        hyperparameters = hyperparameters['params']
        recommender = ItemKNNCBFRecommender(URM_train=train, ICM_train=ICM_dataframe, verbose=False)
        recommender.fit(
            topK=int(hyperparameters['topK']),
            shrink=hyperparameters['shrink'],
            similarity=hyperparameters['similarity'],
            feature_weighting=hyperparameters['feature_weighting']
        )

write_submission(recommender=recommender)
