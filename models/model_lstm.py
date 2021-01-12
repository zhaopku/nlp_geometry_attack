import torch
import os
import numpy as np
from torch import nn
from models.mlp import MLP

class ModelLSTM(nn.Module):
	def __init__(self, args, pre_trained_embedding, vocab_size):
		super(ModelLSTM, self).__init__()
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

		# set batch_first to True
		self.rnn = nn.LSTM(input_size=self.args.embedding_size, hidden_size=self.args.hidden_size,
		              batch_first=True, dropout=0.0, num_layers=1, bidirectional=self.args.bidirectional)

		in_features = self.args.hidden_size*2 if self.args.bidirectional else self.args.hidden_size

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

	def find_last_relevant_output(self, outputs, lengths):
		"""

		:param outputs: [batch_size, max_steps, hidden_size]
		:param lengths: [batch_size]
		:return:
		"""
		_, _, hidden_size = outputs.size()

		# convert length to index
		index = lengths - 1
		index = torch.reshape(index, (-1, 1, 1))
		index = index.repeat(1, 1, hidden_size)
		last_relevant_outputs = torch.gather(outputs, dim=1, index=index)
		last_relevant_outputs = torch.squeeze(last_relevant_outputs, dim=1)
		return last_relevant_outputs

	def build_output_mean(self, outputs, lengths):
		"""

		:param outputs: [batch_size, max_steps, hidden_size]
		:param lengths: [batch_size]
		:return:
		"""
		cur_batch_size = outputs.size(0)
		max_steps = outputs.size(1)

		# [batch_size, max_steps]
		mask = torch.arange(max_steps).view(1, -1).repeat(cur_batch_size, 1)
		if torch.cuda.is_available():
			mask = mask.cuda()
		mask = mask < lengths.unsqueeze(-1)
		# [batch_size, max_steps, 1]
		mask = mask.unsqueeze(-1)

		# [batch_size, max_steps, hidden_size]
		outputs = torch.mul(outputs, mask)
		# [batch_size, hidden_size]
		outputs = outputs.sum(1)/lengths.unsqueeze(-1)

		return outputs

	def forward(self, word_ids, lengths, return_embedded=False):
		"""
		:param word_ids: [batch_size, max_steps]
		:param lengths: [batch_size]
		:param return_embedded:
		:return:
		"""
		# embedded: [batch_size, max_steps, embedding_size]
		# embedded = self.embedding_layer(word_ids.long()).float().data
		#
		# embedded = embedded.clone().data.detach_()
		#
		# embedded.requires_grad_(True)

		embedded = self.embedding_layer(word_ids.long()).float()

		# outputs: [batch_size, max_steps, hidden_size]
		outputs, (h_n, c_n) = self.rnn(embedded)

		last_relevant_outputs = self.build_output_mean(outputs, lengths)

		logits = self.hidden(last_relevant_outputs)

		if return_embedded:
			return logits, last_relevant_outputs, embedded
		else:
			return logits, last_relevant_outputs
