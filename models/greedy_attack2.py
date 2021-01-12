import torch
import numpy as np
import os
from models.attack import AttackLoop
from copy import deepcopy
from attacks.deepfool import DeepFool
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
import string
from time import sleep
from torch.nn import CosineSimilarity
from attacks.word_saliency_batch import WordSaliencyBatch
from nltk.corpus import wordnet
from utils.data_utils import Sample
import sys
import timeit
from timeit import default_timer as timer

class GreedyAttack:
	"""
	Select words greedily as an attack
	"""
	def __init__(self, args, text_data, writer, summary_dir, out):
		self.args = args
		self.text_data = text_data
		self.writer = writer
		self.summary_dir = summary_dir
		self.out = out
		self.stopwords = set(stopwords.words('english'))
		self.mode = None
		self.samples = None
		self.cosine_similarity = CosineSimilarity(dim=1, eps=1e-6)
		self.global_step = 0

		if self.args.attack == 'deepfool':
			# in fact, deepfool will finish far more quicker than this
			self.attack = DeepFool(args=self.args, num_classes=2, max_iters=20)
		else:
			print('Attack {} not recognized'.format(self.args.attack))

		self.model = None
		self.word_saliency = WordSaliencyBatch(args=self.args, text_data=self.text_data)

	def select_word_batch(self, all_word_ids, cur_available, labels, lengths, finish_mask, stopwords_mask, mask, previous_replaced_words=None):
		"""
		select words in a batch fashion
		:param all_word_ids: [batch_size, max_steps]
		:param cur_available: [batch_size, max_steps]
		:param labels: [batch_size]
		:param lengths: [batch_size]
		:param finish_mask: [batch_size]
		:param stopwords_mask: [batch_size, max_steps]
		:param mask: [batch_size, max_steps]
		:param previous_replaced_words: a list of length batch_size
		:return:
		"""

		# currently, batched word_saliency is too mem consuming
		cur_batch_size = cur_available.size(0)
		all_replace_orders = []

		# t = self.select_word(word_ids=all_word_ids, cur_available=cur_available,
		# 									 label=labels, length=lengths)
		if self.args.abandon_stopwords:
			mask = torch.mul(mask, stopwords_mask)

		if torch.cuda.is_available():
			mask = mask.cuda()
			cur_available = cur_available.cuda()
		mask = torch.mul(mask, cur_available)
		mask = mask.bool()

		_, all_replace_orders = self.word_saliency.compute_saliency(model_=self.model, word_ids=all_word_ids,
																	labels=labels, lengths=lengths, mask=mask, order=True)


		# for idx in range(cur_batch_size):
		# 	cur_replace_order = self.select_word(word_ids=all_word_ids[idx], cur_available=cur_available[idx],
		# 										 label=labels[idx], length=lengths[idx],
		# 										 stopwords_mask=stopwords_mask[idx], mask=mask[idx])
		# 	cur_replace_order.detach_()
		# 	all_replace_orders.append(cur_replace_order)
		# all_replace_orders = torch.cat(all_replace_orders, dim=0)

		if torch.cuda.is_available():
			all_replace_orders = all_replace_orders.cuda()


		# [batch_size, max_steps]
		return all_replace_orders

	def select_word(self, word_ids, cur_available, label, length, stopwords_mask, mask):
		"""
		select which word to replace, one at a time
		following pwwc, we select word to replace by
		:param word_ids: [max_steps]
		:param cur_available: [max_steps]
		:param label: [1]
		:param length: [1]
		:param stopwords_mask: [max_steps]
		:param mask: [max_steps]
		:return:
		"""

		# ignore punctuations, stopwords, and out of vocab words
		# create a mask, in which stopwords, oov words, and paddings are omitted
		# only word with a mask value of 1 will be considered

		if self.args.abandon_stopwords:
			mask = torch.mul(mask, stopwords_mask)

		if torch.cuda.is_available():
			mask = mask.cuda()
			cur_available = cur_available.cuda()

		mask = torch.mul(mask, cur_available)
		mask = mask.bool().unsqueeze(0)
		# [1, max_steps]
		word_ids = word_ids.clone().detach().data.unsqueeze(0)
		# [1, 1]
		labels = label.clone().detach().data.unsqueeze(0)
		lengths = length.clone().detach().data.unsqueeze(0)

		if torch.cuda.is_available():
			word_ids = word_ids.cuda()
			labels = labels.cuda()
			lengths = lengths.cuda()
		# [1]
		_, replace_order = self.word_saliency.compute_saliency(model_=self.model, word_ids=word_ids, labels=labels, lengths=lengths, mask=mask, order=True)

		return replace_order

	def construct_new_sample_batch2(self, word_ids, labels, lengths, word_indices, sample_ids, finish_mask):
		"""

		:param word_ids: [batch_size, max_steps]
		:param labels: [batch_size]
		:param lengths: [batch_size]
		:param word_indices: [batch_size], the best word in each example to replace
		:param sample_ids: [batch_size]
		:param finish_mask: [batch_size]
		:return:
		"""
		cur_batch_size = sample_ids.size(0)

		all_new_lengths = []
		all_new_labels = []
		all_new_word_ids = []

		n_new_samples = []

		for idx in range(cur_batch_size):
			new_word_ids, new_lengths, new_labels = self.construct_new_sample2(word_ids=word_ids[idx], label=labels[idx], length=lengths[idx],
													word_idx=word_indices[idx], sample_id=sample_ids[idx], finish_mask=finish_mask[idx])
			all_new_word_ids.append(new_word_ids)
			all_new_lengths.append(new_lengths)
			all_new_labels.append(new_labels)
			n_new_samples.append(new_labels.size(0))

		all_new_word_ids = torch.cat(all_new_word_ids)
		all_new_lengths = torch.cat(all_new_lengths)
		all_new_labels = torch.cat(all_new_labels)

		return all_new_word_ids, all_new_lengths, all_new_labels, n_new_samples

	def construct_new_sample2(self, word_ids, label, length, word_idx, sample_id, finish_mask):
		"""

		:param word_ids:
		:param label:
		:param length:
		:param word_idx:
		:param sample_id:
		:param finish_mask:
		:return: all_new_word_ids, [N, max_steps]
				 all_new_lengths, []
				 all_new_labels
		"""
		all_new_lengths = []
		all_new_labels = []
		all_new_word_ids = []

		if finish_mask:
			word_ids = word_ids.unsqueeze(0)
			length = length.unsqueeze(0)
			label = label.unsqueeze(0)

			return word_ids, length, label

		old_id = int(word_ids[word_idx].data.cpu().numpy())
		syn_word_ids = self.text_data.wordid2synonyms[old_id]

		# if sample_id == 33:
		# 	for i in syn_word_ids:
		# 		w = self.text_data.id2word[i]
		# 		print(w)

		for i in range(len(syn_word_ids)):
			new_id = syn_word_ids[i]
			new_word_ids = deepcopy(word_ids)
			new_word_ids[word_idx] = new_id

			all_new_word_ids.append(new_word_ids)
			all_new_lengths.append(length)
			all_new_labels.append(label)

		all_new_word_ids = torch.stack(all_new_word_ids)
		all_new_lengths = torch.stack(all_new_lengths)
		all_new_labels = torch.stack(all_new_labels)

		return all_new_word_ids, all_new_lengths, all_new_labels

	def construct_new_sample_batch(self, word_ids, labels, lengths, word_indices, sample_ids, finish_mask):
		"""

		:param word_ids: [batch_size, max_steps]
		:param labels: [batch_size]
		:param lengths: [batch_size]
		:param word_indices: [batch_size], the best word in each example to replace
		:param sample_ids: [batch_size]
		:param finish_mask: [batch_size]
		:return:
		"""
		cur_batch_size = sample_ids.size(0)

		all_new_samples = []
		n_new_samples = []

		for idx in range(cur_batch_size):
			new_samples = self.construct_new_sample(word_ids=word_ids[idx], label=labels[idx], length=lengths[idx],
													word_idx=word_indices[idx], sample_id=sample_ids[idx], finish_mask=finish_mask[idx])
			all_new_samples.extend(new_samples)
			n_new_samples.append(len(new_samples))

		return all_new_samples, n_new_samples

	def construct_new_sample(self, word_ids, label, length, word_idx, sample_id, finish_mask):
		"""
		replace the word_idx-th word in self.samples[sample_id] with its synonyms
		:param word_ids:
		:param label:
		:param length:
		:param word_idx:
		:param sample_id:
		:param finish_mask:
		:return:
		"""

		# find the original word
		word = self.text_data.id2word[word_ids[word_idx].cpu().data.item()]
		words = [self.text_data.id2word[word_ids[idx].cpu().data.item()] for idx in range(len(word_ids))]
		if finish_mask:
			cur_sample = self.samples[sample_id]
			if isinstance(cur_sample.length, int):
				cur_sample.id = torch.tensor(cur_sample.id)
			cur_sample.id = cur_sample.id.clone().detach().data
			if isinstance(cur_sample.word_ids, list):
				cur_sample.word_ids = torch.tensor(cur_sample.word_ids)
			cur_sample.word_ids = cur_sample.word_ids[:self.args.max_steps].clone().detach().data
			if cur_sample.length > self.args.max_steps:
				cur_sample.length = self.args.max_steps
			if isinstance(cur_sample.length, int):
				cur_sample.length = torch.tensor(cur_sample.length)
			cur_sample.length = cur_sample.length.clone().detach().data

			if torch.cuda.is_available():
				cur_sample.id = cur_sample.id.cuda()
				cur_sample.word_ids = cur_sample.word_ids.cuda()
				cur_sample.length = cur_sample.length.cuda()

			return [cur_sample]

		synonyms = []
		for syn in wordnet.synsets(word):
			for l in syn.lemmas():
				w = l.name()
				if w not in self.text_data.vocab:
					continue
				synonyms.append(w)
		# put original word in synonyms
		synonyms.append(word)
		synonyms = list(set(synonyms))
		syn_word_ids = [self.text_data.word2id[w] for w in synonyms]


		# if sample_id == 33:
		# 	for i in syn_word_ids:
		# 		w = self.text_data.id2word[i]
		# 		print(w)

		# # this word does not have synonyms at all
		# if len(synonyms) < 2:
		# 	return None

		new_samples = []
		for i in range(len(syn_word_ids)):
			new_id = syn_word_ids[i]
			new_word = synonyms[i]
			old_word = word
			old_id = self.text_data.word2id[old_word]
			# 	def __init__(self, data, words, steps, label, length, id):
			cur_new_sample = Sample(data=word_ids, words=words, steps=self.args.max_steps, label=label, length=length, id=sample_id)
			cur_new_sample.word_ids[word_idx] = new_id
			cur_new_sample.sentence[word_idx] = new_word
			cur_new_sample.set_new_info((new_id, new_word, old_id, old_word, word_idx.cpu().data.item()))
			# record history of changes
			# self.samples[sample_id].history.append((new_id, new_word, old_id, old_word, word_idx.cpu().data.item()))
			new_samples.append(deepcopy(cur_new_sample))

		return new_samples

	def adv_attack(self, word_ids, lengths, labels, sample_ids, model, samples, stopwords_mask, mask):
		"""
		attack a batch of words
		:param word_ids: [batch_size, max_steps]
		:param lengths: [batch_size]
		:param labels: [batch_size]
		:param sample_ids: [batch_size]
		:param model:
		:param samples:
		:param stopwords_mask: [batch_size, max_steps]
		:param mask: [batch_size, max_steps]
		:return:
		"""
		"""

		:param word_ids: [batch_size, max_length]
		:param lengths: [batch_size]
		:param labels: [batch_size]
		:param sample_ids: [batch_size]
		:param model: current model, deepcopy before use
		:param mode:
		:return:
		"""
		# important, set model to eval mode
		self.model = deepcopy(model)
		self.model.eval()
		self.samples = samples

		# self.mode = mode
		# if mode == 'train':
		# 	self.samples = self.text_data.train_samples
		# elif mode == 'val':
		# 	self.samples = self.text_data.val_samples
		# else:
		# 	self.samples = self.text_data.test_samples

		cur_batch_size = word_ids.size(0)

		# logits: [batch_size, num_classes]
		# sent_vecs: [batch_size, hidden_size]
		logits, sent_vecs = model(word_ids, lengths)
		# preds: [batch_size], original predictions before perturbing
		original_predictions = torch.argmax(logits, dim=-1)
		num_classes = logits.size(1)
		# [batch_size, num_classes]
		# select by original prediction
		one_hot_mask = torch.arange(num_classes).unsqueeze(0).repeat(cur_batch_size, 1)
		if torch.cuda.is_available():
			one_hot_mask = one_hot_mask.cuda()
		one_hot_mask = one_hot_mask == original_predictions.unsqueeze(1)

		original_probs = torch.nn.functional.softmax(logits, dim=-1)
		pred_probs = torch.masked_select(original_probs, one_hot_mask)
		intermediate_pred_probs = []
		intermediate_pred_probs.append(pred_probs)

		# # find the boundary point
		# self.model.zero_grad()
		# normals, pert_vecs, all_original_predictions = self.attack(vecs=sent_vecs, net_=model.hidden)
		# # [batch_size, hidden_size]
		# r_tot = pert_vecs - sent_vecs

		cur_available = torch.ones(cur_batch_size, self.args.max_steps)

		# [batch_size]
		finish_mask = torch.zeros(cur_batch_size).bool()
		cur_projections = torch.zeros(cur_batch_size)
		cur_predictions = deepcopy(original_predictions.data)
		# [batch_size, max_steps]
		cur_word_ids = deepcopy(word_ids)
		# [batch_size, hidden_size]
		cur_sent_vecs = deepcopy(sent_vecs.data)

		if torch.cuda.is_available():
			finish_mask = finish_mask.cuda()
			cur_predictions = cur_predictions.cuda()
			cur_projections = cur_projections.cuda()
			cur_word_ids = cur_word_ids.cuda()
			cur_available = cur_available.cuda()
			cur_sent_vecs = cur_sent_vecs.cuda()

		intermediate_projections = []
		intermediate_normals = []
		intermediate_cosines = []
		intermediate_distances = []
		# [batch_size, iter_idx]
		intermediate_word_ids = []
		intermediate_update_masks = []

		# all_word_ids, cur_available, labels, lengths, finish_mask
		# [batch_size, max_steps]

		# all_replace_orders = self.select_word_batch(all_word_ids=word_ids, cur_available=cur_available,
		# 											labels=labels, lengths=lengths, finish_mask=finish_mask,
		# 											stopwords_mask=stopwords_mask, mask=mask)
		previous_replaced_words = []
		intermediate_word_ids.append(word_ids)
		for iter_idx in range(self.args.max_loops):
			# torch.cuda.empty_cache()
			# print('max memory = {}, current memory = {}'.format(torch.cuda.max_memory_allocated()/1024/1024, torch.cuda.memory_allocated()/1024/1024))
			# print('start loop')
			# torch.cuda.reset_max_memory_allocated()
			# torch.cuda.reset_max_memory_cached()
			# print(torch.cuda.max_memory_allocated()/1024/1024)
			# print(torch.cuda.memory_allocated()/1024/1024)

			if finish_mask.sum() == cur_batch_size:
				break

			self.model.zero_grad()
			# for cur_samples, find boundary point
			# cur_normals: [batch_size, hidden_size]
			# cur_pert_vecs: [batch_size, hidden_size]
			# cur_original_predictions: [batch_size]
			# update cur_normals, cur_pert_vecs, cur_original_predictions
			cur_normals, cur_pert_vecs, cur_original_predictions = self.attack(vecs=cur_sent_vecs, net_=self.model.hidden)

			intermediate_normals.append(cur_normals)

			# [batch_size, hidden_size]
			cur_r_tot = cur_pert_vecs - cur_sent_vecs
			# [batch_size], distances to decision boundary
			cur_r_tot_distance = self.norm_dim(cur_r_tot)
			intermediate_distances.append(cur_r_tot_distance)

			# words_to_replace: [batch_size]
			# cur_available: [batch_size, max_steps]
			# cur_available is updated in selected_word_batch
			all_replace_orders = self.select_word_batch(all_word_ids=cur_word_ids, cur_available=cur_available,
														labels=labels, lengths=lengths, finish_mask=finish_mask,
														stopwords_mask=stopwords_mask, mask=mask)
			words_to_replace = all_replace_orders[:, 0]

			words_to_replace_one_hot = torch.nn.functional.one_hot(words_to_replace, num_classes=word_ids.size(1))
			cur_available = torch.mul(cur_available, 1-words_to_replace_one_hot)

			# all_new_samples have N samples inside
			# n_new_samples: [batch_size], number of new samples for each old sample
			# def construct_new_sample_batch(self, word_ids, labels, lengths, word_indices, sample_ids, finish_mask):
			# start = timer()
			all_new_word_ids, all_new_lengths, all_new_labels, n_new_samples = self.construct_new_sample_batch2(word_ids=cur_word_ids,
																			 labels=labels, lengths=lengths,
																			 word_indices=words_to_replace,
																			 sample_ids=sample_ids, finish_mask=finish_mask)

			assert all_new_word_ids.size(0) == all_new_labels.size(0)

			# end = timer()
			# print('new {}'.format(end - start))

			# print('--------')
			# start = timer()
			# all_new_samples, n_new_samples = self.construct_new_sample_batch(word_ids=cur_word_ids,
			# 																 labels=labels, lengths=lengths,
			# 																 word_indices=words_to_replace,
			# 																 sample_ids=sample_ids, finish_mask=finish_mask)
			# end = timer()
			# print('old {}'.format(end - start))

			# assert len(all_new_samples) == np.asarray(n_new_samples).sum()
			#
			# all_new_word_ids = []
			# all_new_lengths = []
			# all_new_labels = []
			# all_new_sentences = []
			#
			# for i, new_sample in enumerate(all_new_samples):
			# 	all_new_word_ids.append(new_sample.word_ids)
			# 	all_new_lengths.append(new_sample.length)
			# 	all_new_labels.append(new_sample.label)
			# 	all_new_sentences.append(new_sample.sentence)
			#
			# # put to GPU
			# all_new_word_ids = torch.stack(all_new_word_ids, dim=0)
			# all_new_lengths = torch.tensor(all_new_lengths)
			# all_new_labels = torch.tensor(all_new_labels)

			if torch.cuda.is_available():
				# [N, max_steps]
				all_new_word_ids = all_new_word_ids.cuda()
				# [N]
				all_new_lengths = all_new_lengths.cuda()
				all_new_labels = all_new_labels.cuda()

			# compute new sent_vecs
			# all_new_logits: [N, num_classes]
			# all_new_sent_vectors: [N, hidden_size]
			all_new_logits, all_new_sent_vectors = model(all_new_word_ids, all_new_lengths)

			# [N]
			all_new_predictions = torch.argmax(all_new_logits, dim=-1)
			# [N, num_classes]
			all_new_probs = torch.softmax(all_new_logits, dim=-1).data

			# get new r_tot
			# [N, hidden_size]
			repeats = torch.tensor(n_new_samples)
			if torch.cuda.is_available():
				repeats = repeats.cuda()

			all_cur_sent_vecs = torch.repeat_interleave(cur_sent_vecs, repeats=repeats, dim=0)
			all_cur_normals = torch.repeat_interleave(cur_normals, repeats=repeats, dim=0)
			all_new_r_tot = all_new_sent_vectors - all_cur_sent_vecs

			# [N]
			all_new_r_tot_length = self.norm_dim(all_new_r_tot)
			all_cosines = self.cosine_similarity(all_new_r_tot, all_cur_normals)
			all_projections = torch.mul(all_new_r_tot_length, all_cosines)

			# TODO: instead of projections, use nearest point
			if self.args.metric != 'projection':
				all_cur_normals, all_cur_pert_vecs, all_cur_original_predictions = self.attack(
					vecs=all_new_sent_vectors, net_=model.hidden)
				# [N, hidden_size]
				all_cur_r_tot = all_cur_pert_vecs - all_cur_sent_vecs

				# [N]
				all_cur_r_tot_distance = self.norm_dim(all_cur_r_tot)
				all_projections = all_cur_r_tot_distance

			# split all_projections to match individual examples
			# list of tensors, list length: [batch_size]
			all_projections_splited = torch.split(all_projections, split_size_or_sections=n_new_samples)
			all_new_predictions_splited = torch.split(all_new_predictions, split_size_or_sections=n_new_samples)
			all_new_lengths_splited = torch.split(all_new_lengths, split_size_or_sections=n_new_samples)
			all_new_labels_splited = torch.split(all_new_labels, split_size_or_sections=n_new_samples)
			all_cosines_splited = torch.split(all_cosines, split_size_or_sections=n_new_samples)

			# list length: [batch_size]
			# each item in the list is a tensor, which consists of several tensors of length max_steps
			all_new_word_ids_splited = torch.split(all_new_word_ids, split_size_or_sections=n_new_samples, dim=0)
			all_new_sent_vectors_splited = torch.split(all_new_sent_vectors, split_size_or_sections=n_new_samples, dim=0)
			all_new_probs_splited = torch.split(all_new_probs, split_size_or_sections=n_new_samples, dim=0)

			# for each tensor, pick the one with largest projection
			assert len(all_projections_splited) == cur_batch_size
			# [batch_size]
			selected_indices = []
			selected_projections = []
			selected_predictions = []
			selected_cosines = []
			# [batch_size, max_steps]
			selected_word_ids = []

			# [batch_size, hidden_size]
			selected_sent_vecs = []

			selected_new_probs = []

			for i in range(cur_batch_size):
				selected_idx = torch.argmax(all_projections_splited[i])
				selected_projection = torch.max(all_projections_splited[i])
				if self.args.metric != 'projection':
					selected_idx = torch.argmin(all_projections_splited[i])
					selected_projection = torch.min(all_projections_splited[i])
				selected_prediction = all_new_predictions_splited[i][selected_idx]
				selected_cosine = all_cosines_splited[i][selected_idx]
				selected_word_ids_for_cur_sample = all_new_word_ids_splited[i][selected_idx]
				selected_sent_vec_for_cur_sample = all_new_sent_vectors_splited[i][selected_idx]
				selected_probs_for_cur_sample = all_new_probs_splited[i][selected_idx]

				selected_indices.append(selected_idx)
				selected_projections.append(selected_projection)
				selected_predictions.append(selected_prediction)
				selected_word_ids.append(selected_word_ids_for_cur_sample)
				selected_sent_vecs.append(selected_sent_vec_for_cur_sample)
				selected_cosines.append(selected_cosine)
				selected_new_probs.append(selected_probs_for_cur_sample)

			# [batch_size]
			selected_indices = torch.tensor(selected_indices)
			selected_projections = torch.tensor(selected_projections)
			selected_predictions = torch.tensor(selected_predictions)
			selected_cosines = torch.tensor(selected_cosines)
			# [batch_size, max_steps]
			selected_word_ids = torch.stack(selected_word_ids, 0)
			# [batch_size, hidden_size]
			selected_sent_vecs = torch.stack(selected_sent_vecs, 0)
			# [batch_size, num_classes]
			selected_new_probs = torch.stack(selected_new_probs, 0)

			# [batch_size]
			cur_pred_probs = torch.masked_select(selected_new_probs, one_hot_mask)
			intermediate_pred_probs.append(cur_pred_probs)

			if torch.cuda.is_available():
				selected_indices = selected_indices.cuda()
				selected_projections = selected_projections.cuda()
				selected_predictions = selected_predictions.cuda()
				selected_word_ids = selected_word_ids.cuda()
				selected_sent_vecs = selected_sent_vecs.cuda()

			# update cur_projections, cur_predictions, and cur_word_ids by ~finish_mask
			# all unfinished samples need to be updated
			# [batch_size]
			cur_update_mask = ~finish_mask
			cur_update_mask = torch.mul(cur_update_mask, selected_projections > 0)

			# torch.where(condition, x, y) → Tensor
			# x if condition else y
			# [batch_size]
			cur_projections = torch.where(cur_update_mask, selected_projections, cur_projections)
			cur_predictions = torch.where(cur_update_mask, selected_predictions, cur_predictions)
			# [batch_size, max_steps]
			cur_word_ids = torch.where(cur_update_mask.view(-1, 1), selected_word_ids, cur_word_ids)
			intermediate_word_ids.append(cur_word_ids)
			# [batch_size, hidden_size]
			cur_sent_vecs = torch.where(cur_update_mask.view(-1, 1), selected_sent_vecs, cur_sent_vecs)

			cur_sent_vecs.detach_()
			cur_word_ids.detach_()
			cur_projections.detach_()
			cur_predictions.detach_()

			intermediate_projections.append(cur_projections.data)
			intermediate_cosines.append(selected_cosines.data)

			# if torch.cuda.is_available():
			# 	print(torch.cuda.max_memory_allocated())
			# 	print(torch.cuda.memory_allocated())
			# 	sys.stdout.flush()

			# finish if we successfully fool the model
			# [batch_size]
			cur_finish_mask = (selected_predictions != original_predictions)
			intermediate_update_masks.append(cur_finish_mask)
			finish_mask += cur_finish_mask
			finish_mask = finish_mask.bool()

		# for the last sent_vec, calculate its distance to decision boundary
		final_normals, final_pert_vecs, final_original_predictions = self.attack(vecs=cur_sent_vecs, net_=model.hidden)
		intermediate_normals.append(final_normals)
		# [batch_size, hidden_size]
		final_r_tot = final_pert_vecs - cur_sent_vecs
		# [batch_size], distances to decision boundary
		final_r_tot_distance = self.norm_dim(final_r_tot)
		intermediate_distances.append(final_r_tot_distance)

		# [batch_size, hidden_size]
		final_r_tot = cur_sent_vecs - sent_vecs
		# [batch_size, max_steps]
		final_word_ids = deepcopy(cur_word_ids)

		# [batch_size]
		final_predictions = deepcopy(cur_predictions)
		# [batch_size, n_loops]
		intermediate_cosines = torch.stack(intermediate_cosines).transpose(0, 1)
		intermediate_projections = torch.stack(intermediate_projections).transpose(0, 1)
		# [batch_size, n_loops+1]
		intermediate_distances = torch.stack(intermediate_distances).transpose(0, 1)
		intermediate_pred_probs = torch.stack(intermediate_pred_probs).transpose(0, 1)

		# [batch_size, loops+1, max_steps]
		intermediate_word_ids = torch.stack(intermediate_word_ids).transpose(0, 1)
		intermediate_normals = torch.stack(intermediate_normals).transpose(0, 1)

		if torch.cuda.is_available():
			final_r_tot = final_r_tot.cuda()
			final_word_ids = final_word_ids.cuda()
			final_predictions = final_predictions.cuda()
			intermediate_normals = intermediate_normals.cuda()
			intermediate_cosines = intermediate_cosines.cuda()
			intermediate_projections = intermediate_projections.cuda()
			intermediate_distances = intermediate_distances.cuda()
		# print('n_loops = {}'.format(iter_idx))
		return final_r_tot, final_word_ids, final_predictions, intermediate_normals,\
			   intermediate_cosines, intermediate_distances, original_predictions, intermediate_word_ids, intermediate_pred_probs

	@staticmethod
	def norm_dim(w):
		norms = []
		for idx in range(w.size(0)):
			norms.append(w[idx].norm())
		norms = torch.stack(tuple(norms), dim=0)

		return norms
