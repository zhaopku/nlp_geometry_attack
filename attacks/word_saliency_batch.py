import torch
import os
import tqdm
import nltk
import numpy
from copy import deepcopy
from torch import nn
from timeit import default_timer as timer
import sys

class WordSaliencyBatch:
	def __init__(self, args, text_data):
		self.args = args
		self.text_data = text_data

		self.model = None

	def split_forward(self, new_word_ids, new_lengths):

		# split new_word_ids and new_lengths
		new_word_ids_splits = new_word_ids.split(self.args.splits, dim=0)
		new_lengths_splits = new_lengths.split(self.args.splits, dim=0)

		new_logits = []
		for idx in range(len(new_lengths_splits)):

			new_logits_split, _ = self.model(new_word_ids_splits[idx], new_lengths_splits[idx])
			new_logits.append(new_logits_split)

		new_logits = torch.cat(new_logits, dim=0)
		return new_logits

	def compute_saliency(self, model_, word_ids, labels, lengths, mask, order=False):
		"""
		compute saliency for a batch of examples
		# TODO: implement batch to more than one examples
		:param model_:
		:param word_ids: [batch_size, max_steps]
		:param labels: [batch_size]
		:param lengths: [batch_size]
		:param mask: [batch_size, max_steps]
		:param order:
		:return:
		"""
		with torch.no_grad():
			# print mem usage
			# print('start')
			# torch.cuda.reset_max_memory_allocated()
			# torch.cuda.reset_max_memory_cached()
			# print(torch.cuda.max_memory_allocated()/1024/1024)
			# print(torch.cuda.memory_allocated()/1024/1024)

			self.model = deepcopy(model_)
			if self.args.model == 'lstm':
				self.model.rnn.flatten_parameters()
			# self.model = nn.DataParallel(self.model)
			self.model.eval()
			# self.model = model_
			cur_batch_size = word_ids.size(0)

			unk_id = self.text_data.word2id[self.text_data.UNK_WORD]
			unk_id = torch.tensor(unk_id)
			if torch.cuda.is_available():
				unk_id = unk_id.cuda()

			# compute the original probs for true class
			# logits: [batch_size, num_classes]
			# predictions: [batch_size]
			# probs: [batch_size, num_classes]
			logits, _ = self.model(word_ids, lengths)
			predictions = torch.argmax(logits, dim=-1)
			probs = torch.softmax(logits, dim=-1)

			# [batch_size, num_classes]
			one_hot_mask = torch.arange(logits.size(1)).unsqueeze(0).repeat(cur_batch_size, 1)
			if torch.cuda.is_available():
				one_hot_mask = one_hot_mask.cuda()

			one_hot_mask = one_hot_mask == predictions.unsqueeze(1)

			# [batch_size, 1]
			true_probs = torch.masked_select(probs, one_hot_mask)

			# unsqueeze word_ids
			# [batch_size, 1, max_steps]
			new_word_ids = word_ids.unsqueeze(1)
			# [batch_size, max_steps, max_steps]
			# dim 1 used to indicate which word is replaced by unk
			new_word_ids = new_word_ids.repeat(1, self.args.max_steps, 1)

			# then replace word by unk
			# [max_steps, max_steps]
			# diagonal elements = 1
			diag_mask = torch.diag(torch.ones(self.args.max_steps))

			# [1, max_steps, max_steps]
			diag_mask = diag_mask.unsqueeze(0)

			# [batch_size, max_steps, max_steps]
			# for elements with a mask of 1, replace with unk_id
			diag_mask = diag_mask.repeat(cur_batch_size, 1, 1).bool()
			if torch.cuda.is_available():
				diag_mask = diag_mask.cuda()

			# [batch_size, max_steps, max_steps]
			# replace with unk_id
			new_word_ids = diag_mask * unk_id + (~diag_mask) * new_word_ids


			# compute probs for new_word_ids
			# [batch_size*max_steps, max_steps]
			new_word_ids = new_word_ids.view(cur_batch_size*self.args.max_steps, -1)

			# construct new_lengths
			# [batch_size, 1]
			new_lengths = lengths.view(cur_batch_size, 1)
			# repeat
			# [batch_size*max_steps]
			new_lengths = new_lengths.repeat(1, self.args.max_steps).view(-1)

			# the same applies to new_predictions
			# [batch_size, 1]
			new_predictions = predictions.view(cur_batch_size, 1)
			# repeat
			# [batch_size*max_steps]
			new_predictions = new_predictions.repeat(1, self.args.max_steps).view(-1)

			# [batch_size*max_steps, num_classes]
			one_hot_mask = torch.arange(logits.size(1)).unsqueeze(0).repeat(new_predictions.size(0), 1)
			if torch.cuda.is_available():
				one_hot_mask = one_hot_mask.cuda()

			one_hot_mask = one_hot_mask == new_predictions.unsqueeze(1)

			# print mem usage
			# print('middle')
			# print(torch.cuda.max_memory_allocated()/1024/1024)
			# print(torch.cuda.memory_allocated()/1024/1024)

			# [batch_size*max_steps, num_classes]
			# new_logits, _ = self.model(new_word_ids, new_lengths)
			# start = timer()
			new_logits = self.split_forward(new_word_ids, new_lengths)
			# end = timer()
			# print('time = {}'.format(end - start))
			sys.stdout.flush()
			# [batch_size*max_steps, num_classes]
			all_probs = torch.softmax(new_logits, dim=-1)

			# print mem usage
			# print('end')
			# print(torch.cuda.max_memory_allocated()/1024/1024)
			# print(torch.cuda.memory_allocated()/1024/1024)

			# [batch_size, max_steps]
			all_true_probs = torch.masked_select(all_probs, one_hot_mask).view(cur_batch_size, -1)

			# only words with a mask of 1 will be considered
			# setting the prob of unqualified words to a large number
			all_true_probs[~mask] = 100.0

			if torch.cuda.is_available():
				all_true_probs = all_true_probs.cuda()

			# [batch_size, max_steps]
			saliency = true_probs.unsqueeze(1) - all_true_probs

			# select the word with the largest saliency
			# [batch_size]
			best_word_idx = torch.argmax(saliency, dim=1)
			replace_order = torch.argsort(saliency, descending=True)

			# check
			check = (best_word_idx < lengths).sum().data.cpu().numpy()

			# assert check == cur_batch_size

			if order:
				return best_word_idx, replace_order
			else:
				return best_word_idx



