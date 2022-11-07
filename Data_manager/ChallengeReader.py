import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager


def _loadURM(URM_path, separator=','):

	URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=separator, dtype={0: int, 1: int, 2: str, 3: int}, engine='python')
	URM_all_dataframe.rename(columns={URM_all_dataframe.columns[0]: 'user_id',
									  URM_all_dataframe.columns[1]: 'item_id',
									  URM_all_dataframe.columns[2]: 'impressions',
									  URM_all_dataframe.columns[3]: 'data'},
							 inplace=True)
	URM_all_dataframe.columns = ["user_id", "item_id", "impressions", "data"]

	URM_impressions_dataframe = URM_all_dataframe.copy().drop(columns=["impressions"])
	URM_all_dataframe = URM_all_dataframe.drop(columns=["impressions"])
	URM_impressions_dataframe.columns = ["user_id", "item_id", "data"]
	URM_all_dataframe.columns = ["user_id", "item_id", "data"]

	return URM_all_dataframe, URM_impressions_dataframe


def _load_ICMs(ICM_type_path, ICM_length_path):

	ICM_type_dataframe = pd.read_csv(filepath_or_buffer=ICM_type_path, dtype={0: str, 1: str, 2: str}, engine='python')
	ICM_length_dataframe = pd.read_csv(filepath_or_buffer=ICM_length_path, dtype={0: str, 1: str, 2: str}, engine='python')

	ICM_type_dataframe.columns = ["item_id", "feature_id", "data"]
	ICM_length_dataframe.columns = ["item_id", "feature_id", "data"]

	# Split GenreList in order to obtain a dataframe with a tag per row

	return ICM_type_dataframe, ICM_length_dataframe


class ChallengeReader(DataReader):

	DATASET_FOLDER = "../data"
	AVAILABLE_URM = ["URM_all"]
	AVAILABLE_ICM = ["ICM_length", "ICM_type"]

	IS_IMPLICIT = True

	def _get_dataset_name_root(self):
		return self.DATASET_FOLDER


	def _load_from_original_file(self):
		# Load data from original

		ICM_type_path = "../data/data_ICM_type.csv"
		ICM_length_path = "../data/data_ICM_length.csv"
		URM_path = "../data/interactionScored.csv"

		self._print("Loading Interactions")
		URM_all_dataframe, URM_impressions_dataframe = _loadURM(URM_path)

		self._print("Loading Item Features type")
		ICM_type_dataframe, ICM_length_dataframe = _load_ICMs(ICM_type_path, ICM_length_path)

		dataset_manager = DatasetMapperManager()
		dataset_manager.add_URM(URM_all_dataframe, "URM_all")
		dataset_manager.add_ICM(ICM_type_dataframe, "ICM_type")
		dataset_manager.add_ICM(ICM_length_dataframe, "ICM_length")

		loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
														  is_implicit=self.IS_IMPLICIT)

		self._print("Loading Complete")

		return loaded_dataset

