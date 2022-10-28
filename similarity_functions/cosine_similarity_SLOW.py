import numpy as np
import scipy.sparse as sp


def cosine_similarity(urm: sp.csc_matrix, shrink: int):
	item_weights = np.sqrt(
		np.sum(urm.power(2), axis=0)
	).A

	numerator = urm.T.dot(urm)
	denominator = item_weights.T.dot(item_weights) + shrink + 1e-6
	weights = numerator / denominator
	np.fill_diagonal(weights, 0.0)

	return weights
