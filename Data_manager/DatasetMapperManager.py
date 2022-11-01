"""
Created on 19/06/2020

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.Dataset import Dataset
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from pandas.api.types import is_string_dtype
import pandas as pd


def _add_keys_to_mapper(key_to_value_mapper, new_key_list):

    for new_key in new_key_list:

        if new_key not in key_to_value_mapper:
            new_value = len(key_to_value_mapper)
            key_to_value_mapper[new_key] = new_value

    return key_to_value_mapper



class DatasetMapperManager(object):
    """
    This class is used to build a Dataset object
    The DatasetMapperManager object takes as input the original data in dataframes.
    The required columns are:
    - URM: "user_id", "item_id", "data"
    - ICM: "item_id", "feature_id", "data"
    - UCM: "user_id", "feature_id", "data"

    The data type of the "data" columns can be any, the "item_id", "user_id", "feature_id" data types MUST be strings.
    How to use it:
    - First add all the necessary data calling the add_URM, add_ICM, add_UCM functions
    - Then call the generate_Dataset function(dataset_name, is_implicit) to obtain the Dataset object.

    The generate_Dataset function will first transform all "item_id", "user_id", "feature_id" into unique numerical indices and
    represent all of them as sparse matrices: URM, ICM, UCM.
    """

    URM_DICT = None
    URM_mapper_DICT = None

    ICM_DICT = None
    ICM_mapper_DICT = None

    UCM_DICT = None
    UCM_mapper_DICT = None

    user_original_ID_to_index = None
    item_original_ID_to_index = None

    __Dataset_finalized = False


    def __init__(self):
        super(DatasetMapperManager, self).__init__()

        self.URM_DICT = {}
        self.URM_mapper_DICT = {}

        self.ICM_DICT = {}
        self.ICM_mapper_DICT = {}

        self.UCM_DICT = {}
        self.UCM_mapper_DICT = {}

        self.__Dataset_finalized = False




    def generate_Dataset(self, dataset_name, is_implicit):

        assert not self.__Dataset_finalized, "Dataset mappers have already been generated, adding new data is forbidden"
        self.__Dataset_finalized = True

        # Generate ID to index mappers
        self._generate_global_mappers()
        self._generate_ICM_UCM_mappers()

        URM_DICT_sparse = {}
        ICM_DICT_sparse = {}
        UCM_DICT_sparse = {}

        on_new_ID = "ignore"

        for URM_name, URM_dataframe in self.URM_DICT.items():
            URM_sparse_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index,
                                                                    preinitialized_row_mapper = self.user_original_ID_to_index,
                                                                    on_new_col = on_new_ID, on_new_row = on_new_ID)

            URM_sparse_builder.add_data_lists(URM_dataframe["user_id"].values,
                                              URM_dataframe["item_id"].values,
                                              URM_dataframe["data"].values)
            URM_DICT_sparse[URM_name] = URM_sparse_builder.get_SparseMatrix()


        for ICM_name, ICM_dataframe in self.ICM_DICT.items():
            feature_ID_to_index = self.ICM_mapper_DICT[ICM_name]
            ICM_sparse_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = feature_ID_to_index,
                                                                    preinitialized_row_mapper = self.item_original_ID_to_index,
                                                                    on_new_col = on_new_ID, on_new_row = on_new_ID)

            ICM_sparse_builder.add_data_lists(ICM_dataframe["item_id"].values,
                                              ICM_dataframe["feature_id"].values,
                                              ICM_dataframe["data"].values)
            ICM_DICT_sparse[ICM_name] = ICM_sparse_builder.get_SparseMatrix()


        for UCM_name, UCM_dataframe in self.UCM_DICT.items():
            feature_ID_to_index = self.UCM_mapper_DICT[UCM_name]
            UCM_sparse_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = feature_ID_to_index,
                                                                    preinitialized_row_mapper = self.user_original_ID_to_index,
                                                                    on_new_col = on_new_ID, on_new_row = on_new_ID)

            UCM_sparse_builder.add_data_lists(UCM_dataframe["user_id"].values,
                                              UCM_dataframe["feature_id"].values,
                                              UCM_dataframe["data"].values)
            UCM_DICT_sparse[UCM_name] = UCM_sparse_builder.get_SparseMatrix()


        loaded_dataset = Dataset(dataset_name=dataset_name,
                                 URM_dictionary=URM_DICT_sparse,
                                 ICM_dictionary=ICM_DICT_sparse,
                                 ICM_feature_mapper_dictionary=self.ICM_mapper_DICT,
                                 UCM_dictionary=UCM_DICT_sparse,
                                 UCM_feature_mapper_dictionary=self.UCM_mapper_DICT,
                                 user_original_ID_to_index=self.user_original_ID_to_index,
                                 item_original_ID_to_index=self.item_original_ID_to_index,
                                 is_implicit=is_implicit,
                                 )

        return loaded_dataset


    def _generate_global_mappers(self):
        """
        Generates the user_id and item_id mapper including all data available: URM, ICM, UCM
        :return:
        """

        self.user_original_ID_to_index = {}
        self.item_original_ID_to_index = {}

        for _, URM_dataframe in self.URM_DICT.items():
            self.user_original_ID_to_index = _add_keys_to_mapper(self.user_original_ID_to_index, URM_dataframe["user_id"].values)
            self.item_original_ID_to_index = _add_keys_to_mapper(self.item_original_ID_to_index, URM_dataframe["item_id"].values)

        for _, ICM_dataframe in self.ICM_DICT.items():
            self.item_original_ID_to_index = _add_keys_to_mapper(self.item_original_ID_to_index, ICM_dataframe["item_id"].values)

        for _, UCM_dataframe in self.UCM_DICT.items():
            self.user_original_ID_to_index = _add_keys_to_mapper(self.user_original_ID_to_index, UCM_dataframe["user_id"].values)


    def _generate_ICM_UCM_mappers(self):
        """
        Generates the feature_id mapper of each ICM and UCM
        :return:
        """

        for ICM_name, ICM_dataframe in self.ICM_DICT.items():
            feature_ID_to_index = _add_keys_to_mapper({}, ICM_dataframe["feature_id"].values)
            self.ICM_mapper_DICT[ICM_name] = feature_ID_to_index

        for UCM_name, UCM_dataframe in self.UCM_DICT.items():
            feature_ID_to_index = _add_keys_to_mapper({}, UCM_dataframe["feature_id"].values)
            self.UCM_mapper_DICT[UCM_name] = feature_ID_to_index




    def add_URM(self, URM_dataframe:pd.DataFrame, URM_name):
        """
        Adds the URM_dataframe to the current dataset object
        :param URM_dataframe:   Expected columns: user_id, item_id, data
        :param URM_name:        String with the name of the URM
        :return:
        """
        assert set(["user_id", "item_id", "data"]).issubset(set(URM_dataframe.columns)), "Dataframe columns not correct"
        assert all(is_string_dtype(URM_dataframe[ID_column]) for ID_column in ["user_id", "item_id"]), "ID columns must be strings"
        assert not self.__Dataset_finalized, "Dataset mappers have already been generated, adding new data is forbidden"
        assert URM_name not in self.URM_DICT, "URM_name alredy exists"
        self.URM_DICT[URM_name] = URM_dataframe


    def add_ICM(self, ICM_dataframe:pd.DataFrame, ICM_name):
        """
        Adds the ICM_dataframe to the current dataset object
        :param ICM_dataframe:   Expected columns: item_id, feature_id, data
        :param ICM_name:        String with the name of the ICM
        :return:
        """
        assert set(["item_id", "feature_id", "data"]).issubset(set(ICM_dataframe.columns)), "Dataframe columns not correct"
        assert all(is_string_dtype(ICM_dataframe[ID_column]) for ID_column in ["item_id", "feature_id"]), "ID columns must be strings"
        assert not self.__Dataset_finalized, "Dataset mappers have already been generated, adding new data is forbidden"
        assert ICM_name not in self.ICM_DICT, "ICM_name alredy exists"
        self.ICM_DICT[ICM_name] = ICM_dataframe



    def add_UCM(self, UCM_dataframe:pd.DataFrame, UCM_name):
        """
        Adds the UCM_dataframe to the current dataset object
        :param UCM_dataframe:   Expected columns: user_id, feature_id, data
        :param UCM_name:        String with the name of the UCM
        :return:
        """
        assert set(["user_id", "feature_id", "data"]).issubset(set(UCM_dataframe.columns)), "Dataframe columns not correct"
        assert all(is_string_dtype(UCM_dataframe[ID_column]) for ID_column in ["user_id", "feature_id"]), "ID columns must be strings"
        assert not self.__Dataset_finalized, "Dataset mappers have already been generated, adding new data is forbidden"
        assert UCM_name not in self.UCM_DICT, "UCM_name alredy exists"
        self.UCM_DICT[UCM_name] = UCM_dataframe








