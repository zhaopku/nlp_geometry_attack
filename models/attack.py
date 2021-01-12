import torch
import torchvision
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from attacks.deepfool import DeepFool
from nltk.corpus import wordnet
import nltk
from nltk.corpus import stopwords
import string
import os
from time import sleep
from torch.nn import CosineSimilarity

class AttackLoop:
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
			self.attack = DeepFool(args=self.args, num_classes=2, max_iters=20)
		else:
			print('Attack {} not recognized'.format(self.args.attack))

	def construct_new_sample(self, sample_id):
		sample = self.samples[sample_id]
		pos_tags = nltk.pos_tag(sample.sentence)
		new_samples = []
		for idx, word in enumerate(sample.sentence):
			if idx >= sample.length:
				break
			cur_new_samples = []

			# ignore punctuations, stopwords, and out of vocab words
			if word in string.punctuation or word.lower() in self.stopwords or word not in self.text_data.word2id.keys():
				continue

			synonyms = []
			antonyms = []
			for syn in wordnet.synsets(word):
				for l in syn.lemmas():
					w = l.name()
					if w not in self.text_data.word2id.keys():
						continue
					synonyms.append(w)
					if l.antonyms():
						antonyms.append(l.antonyms()[0].name())
			# put original word in synonyms
			synonyms.append(word)
			synonyms = list(set(synonyms))
			new_word_ids = [self.text_data.word2id[w] for w in synonyms]
			antonyms = list(set(antonyms))

			if len(synonyms) < 2:
				continue

			for i in range(len(new_word_ids)):
				new_id = new_word_ids[i]
				new_word = synonyms[i]
				old_word = sample.sentence[idx]
				old_id = self.text_data.word2id[old_word]

				cur_new_sample = deepcopy(sample)
				cur_new_sample.word_ids[idx] = new_id
				cur_new_sample.sentence[idx] = new_word
				cur_new_sample.set_new_info((new_id, new_word, old_id, old_word, idx))
				cur_new_samples.append(cur_new_sample)
			# if len(cur_new_samples) < 10:
			# 	continue
			new_samples.append(cur_new_samples)
			# limit the number of new samples for simplicity
			# if len(new_samples) > 100:
			# 	break
		return new_samples

	def adv_attack(self, word_ids, lengths, labels, sample_ids, model, mode):
		"""

		:param word_ids: [batch_size, max_length]
		:param lengths: [batch_size]
		:param labels: [batch_size]
		:param sample_ids: [batch_size]
		:param model: current model, deepcopy before use
		:param mode:
		:return:
		"""
		self.mode = mode
		if mode == 'train':
			self.samples = self.text_data.train_samples
		elif mode == 'val':
			self.samples = self.text_data.val_samples
		else:
			self.samples = self.text_data.test_samples

		cur_batch_size = word_ids.size(0)

		logits, sent_vecs = model(word_ids, lengths)
		preds = torch.argmax(logits, dim=-1)

		# first attack to find the normal and the boundary point
		# we do not care about network parts that are time relevant

		# grad: normal vector of decision, [batch_size, hidden_size]
		# pert_vecs: boundary point, [batch_size, hidden_size]
		# all_original_predictions: predictions based on the unperturbed sentence vectors
		normals, pert_vecs, all_original_predictions = self.attack(vecs=sent_vecs, net_=model.hidden)
		# [batch_size, hidden_size]
		r_tot = pert_vecs - sent_vecs

		logits = model.hidden(pert_vecs)
		all_perturbed_predictions = torch.argmax(logits, -1)
		changed = (all_perturbed_predictions != all_original_predictions)
		all_original_corrects = (all_original_predictions == labels)
		all_perturbed_corrects = (all_perturbed_predictions == labels)

		n_changed = changed.sum()
		n_original_corrects = all_original_corrects.sum()
		n_perturbed_corrects = all_perturbed_corrects.sum()

		# then loop over words to find the best one to replace
		# as we do not have a word replacing strategy at the moment, the only way possible is to loop over words one by one
		all_cos_changed = []
		all_cos_unchanged = []
		all_length_ratio_changed = []
		all_length_ratio_unchanged = []

		# for a single sample
		for idx in range(cur_batch_size):
			if idx == 15:
				break

			cur_word_ids = word_ids[idx]
			cur_length = lengths[idx]
			cur_label = labels[idx]
			cur_sample_id = sample_ids[idx]
			cur_sent_vec = sent_vecs[idx]
			cur_normal = normals[idx]
			original_prediction = all_original_predictions[idx]
			cur_r_tot = r_tot[idx]

			# add cur_pert_vec
			cur_pert_vec = pert_vecs[idx].unsqueeze(0)

			# then iterate to replace

			# list of lists, each sub-list corresponds to replacing one specific word with its synonyms
			new_samples = self.construct_new_sample(sample_id=cur_sample_id)

			# for each word and its synonyms in the sample
			for cur_new_samples in new_samples:
				# get new word_ids as input to neural network
				cur_label = cur_new_samples[0].label
				new_word_ids = [s.word_ids for s in cur_new_samples]
				new_word_ids = torch.tensor(new_word_ids)

				new_lengths = torch.tensor(cur_new_samples[0].length).repeat(len(cur_new_samples))
				if torch.cuda.is_available():
					new_word_ids = new_word_ids.cuda()
					new_lengths = new_lengths.cuda()

				new_logits, new_sent_vectors = model(new_word_ids, new_lengths)

				cur_predictions = torch.argmax(new_logits, dim=-1)

				cur_probs = torch.softmax(new_logits, dim=-1).data.cpu().numpy()

				# not interesting if no predictions have been changed
				cur_corrects = (cur_predictions == cur_label)
				cur_changed = (cur_predictions != original_prediction).cpu().data.numpy()
				n_cur_changed = cur_changed.sum()

				# at least one prediction has to be changed
				# we are not interested in other examples
				if n_cur_changed < 1:
					continue

				# now examine the cosine values
				# which perturbation has the most projection on normal?
				new_r_tot = new_sent_vectors - cur_sent_vec.unsqueeze(0)
				cosines = self.cosine_similarity(new_r_tot, cur_normal.unsqueeze(0))
				length_ratio = self.norm_dim(new_r_tot)/cur_r_tot.norm()

				# what do we want to log?
				# (1) for each sample, changed from what word to what word
				# (2) for each sample, its perturbation ratio
				# (3) for each sample, its cosine value
				# (4) add cur_per_vec, which is the boundary point

				# construct metadata
				metadata = []
				for i, cur_new_sample in enumerate(cur_new_samples):
					ground_truth_prob = cur_probs[i][cur_label]
					if cur_new_sample.new_info[3] == cur_new_sample.new_info[1]:
						cur_metadata = 'original_point, {}, pred = {}, original pred = {}'.format(cur_corrects[i], cur_predictions[i], original_prediction)
					else:
						cur_metadata = '{}, {} -> {} ({}/{}), pred = {}, o_pred = {},' \
									   ' tprob = {:.5f}, ratio = {:.5f}, cos = {:.5f}'.format(cur_corrects[i], cur_new_sample.new_info[3],
																					 cur_new_sample.new_info[1], cur_new_sample.new_info[4],
																					 cur_new_sample.length, cur_predictions[i], original_prediction,
																					ground_truth_prob, length_ratio[i], cosines[i])
					metadata.append(cur_metadata)

				metadata.append('boundary_point')
				# print(new_sent_vectors.size())
				# print(cur_pert_vec.size())
				new_sent_vectors = torch.cat([new_sent_vectors, cur_pert_vec], dim=0)
				self.writer.add_embedding(new_sent_vectors, metadata=metadata, global_step=self.global_step)
				# self.writer.add_embedding(cur_pert_vec.unsqueeze(0), metadata=['boundary point'], global_step=self.global_step)

				self.global_step += 1

				# prepare some statistics
				cur_changed = torch.tensor(cur_changed)
				if torch.cuda.is_available():
					cur_changed = cur_changed.cuda()
				cos_changed = torch.masked_select(cosines, mask=cur_changed)
				cos_unchanged = torch.masked_select(cosines, mask=~cur_changed)
				length_ratio_changed = torch.masked_select(length_ratio, mask=cur_changed)
				length_ratio_unchanged = torch.masked_select(length_ratio, mask=~cur_changed)

				all_cos_changed.extend(cos_changed.data.cpu().numpy().tolist())
				all_cos_unchanged.extend(cos_unchanged.data.cpu().numpy().tolist())
				all_length_ratio_changed.extend(length_ratio_changed.data.cpu().numpy().tolist())
				all_length_ratio_unchanged.extend(length_ratio_unchanged.data.cpu().numpy().tolist())

		return all_cos_changed, all_cos_unchanged, all_length_ratio_changed, all_length_ratio_unchanged

	@staticmethod
	def norm_dim(w):
		norms = []
		for idx in range(w.size(0)):
			norms.append(w[idx].norm())
		norms = torch.stack(tuple(norms), dim=0)

		return norms
