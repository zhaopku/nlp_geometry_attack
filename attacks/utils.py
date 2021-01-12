import torch
import numpy as np
from torch.nn import Embedding

class Candidate:
	def __init__(self):
		# list of word ids
		self.word_ids = []
		self.original_word_ids = []
		# list of tuples, (index, original_word_id, new_word_id)
		self.replacements = []
		# distance to the decision boundary
		self.distance = [ ]

def norm_dim(w):
	norms = []
	for idx in range(w.size(0)):
		norms.append(w[idx].norm())
	norms = torch.stack(tuple(norms), dim=0)

	return norms
