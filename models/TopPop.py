from Utils.data_loader import read_data
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

import numpy as np
import scipy.sparse as sp
from similarity_functions.cosine_similarity_SLOW import cosine_similarity
from typing import Optional

# Hyperparameter
test_split = 0.2
validation_split = 0.1

data_splitter = DataSplitter_leave_k_out('')


