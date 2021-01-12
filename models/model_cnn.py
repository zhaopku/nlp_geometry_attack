import torch
import os
import numpy as np
from torch import nn
from models.mlp import MLP

class ModelCNN(nn.Module):
	def __init__(self, args, pre_trained_embedding, vocab_size):
		super(ModelCNN, self).__init__()
		self.args = args
		self.pre_trained_embedding = pre_trained_embedding
		self.vocab_size = vocab_size

		if self.args.embedding.startswith('glove'):
			self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(self.pre_trained_embedding), freeze=self.args.freeze_embedding)
		elif self.args.embedding == 'random':
			self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.args.embedding_size)
		else:
			print('{} embedding mode not recognized'.format(self.args.embedding))
			raise NotImplemented

		# update embeddings during training
		self.embedding_layer.weight.requires_grad_(True)

		# TODO: add conv layer here
		self.conv1d = nn.Conv1d(in_channels=self.args.embedding_size, out_channels=self.args.hidden_size, kernel_size=3)

		in_features = self.args.hidden_size

		if self.args.dataset == 'imdb':
			out_features = 2
		elif self.args.dataset == 'agnews':
			out_features = 4
		elif self.args.dataset == 'yahoo':
			out_features = 10
		elif self.args.dataset == 'dbpedia':
			out_features = 14
		else:
			out_features = -1
			print('Unrecognized dataset: {}'.format(self.args.dataset))
			exit(-1)

		self.hidden = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

		# self.hidden = MLP(args=args, in_size=self.args.hidden_size, mlp_size=self.args.mlp_size, out_size=out_features)

	def forward(self, word_ids, lengths, return_embedded=False):
		"""

		:param word_ids:
		:param lengths:
		:param return_embedded:
		:return:
		"""
		# [batch_size, max_steps, embedding_size]
		embedded = self.embedding_layer(word_ids.long()).float()

		# [batch_size, out_steps, hidden_size]
		out_vecs = self.conv1d(embedded.transpose(1, 2)).transpose(1, 2)

		# max pooling
		# [batch_size, hidden_size]
		hidden_vec, _ = torch.max(out_vecs, dim=1)

		# [batch_size, 2]
		logits = self.hidden(hidden_vec)

		if return_embedded:
			return logits, hidden_vec, embedded
		else:
			return logits, hidden_vec
