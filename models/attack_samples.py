import torch
import os
import numpy as np
from models.greedy_attack2 import GreedyAttack
from utils.imdb_data import MyDataSet
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
import sys
import pickle as p

class AttackSamples:
	def __init__(self, args, text_data, writer, summary_dir, out, samples, model):
		self.args = args
		self.text_data = text_data
		self.writer = writer
		self.summary_dir = summary_dir + '_log'
		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)
		self.out = out
		self.samples = samples
		self.model = deepcopy(model)

		self.n_samples_to_disk = self.args.n_samples_to_disk

		self.greedy_attack = GreedyAttack(args=self.args, text_data=self.text_data, writer=self.writer, summary_dir=self.summary_dir, out=self.out)
		self.samples_set = MyDataSet(samples=self.samples, max_steps=self.args.max_steps)
		self.loader = DataLoader(dataset=self.samples_set, num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)

	def attack_all_samples(self):
		self.model.eval()

		if torch.cuda.is_available():
			self.model.cuda()

		results = {'original_acc': 0.0, 'acc_perturbed': 0.0, 'change_rate': 0.0, 'n_samples': 0,
				   'original_corrects': 0, 'perturbed_corrects:': 0, 'n_changed': 0, 'n_perturbed': 0}
		all_replace_rate = []
		all_n_change_words = []
		for idx, (sample_ids, word_ids, lengths, labels, stopwords_mask, mask) in enumerate(tqdm(self.loader)):
			# if idx == 15:
			# 	break
			# torch.cuda.empty_cache()
			# print('batch, max memory = {}, current memory = {}'.format(torch.cuda.max_memory_allocated()/1024/1024, torch.cuda.memory_allocated()/1024/1024))
			cur_batch_size = lengths.size(0)
			if torch.cuda.is_available():
				word_ids = word_ids.cuda()
				lengths = lengths.cuda()
				labels = labels.cuda()
				stopwords_mask = stopwords_mask.cuda()
				sample_ids = sample_ids.cuda()
				mask = mask.cuda()

			logits, _ = self.model(word_ids, lengths)
			predictions = torch.argmax(logits, -1)
			original_corrects = (predictions == labels).float().sum().data.cpu().numpy()
			correct_mask = (predictions == labels)

			if self.args.perturb_correct:
				word_ids = torch.masked_select(word_ids, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
				lengths = torch.masked_select(lengths, correct_mask)
				labels = torch.masked_select(labels, correct_mask)
				stopwords_mask = torch.masked_select(stopwords_mask, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
				mask = torch.masked_select(mask, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
				sample_ids = torch.masked_select(sample_ids, correct_mask)

			results['n_perturbed'] += mask.size(0)
			# [batch_size]
			# perturbed_samples: list of samples
			# perturbed_loops: list of ints
			# perturbed_predictions: tensor
			# original_predictions: tensor
			# perturbed_projections: tensor
			final_r_tot, final_word_ids, perturbed_predictions, intermediate_normals, intermediate_cosines,\
			intermediate_distances, original_predictions, intermediate_word_ids, intermediate_pred_probs = \
				self.greedy_attack.adv_attack(word_ids=word_ids, lengths=lengths, labels=labels,
											  sample_ids=sample_ids, model=self.model, samples=self.samples,
											  stopwords_mask=stopwords_mask, mask=mask)

			# t_logits, _ = self.model(final_word_ids, lengths)
			# t_pred = torch.argmax(t_logits, -1)
			#
			# perturbed_corrects
			perturbed_corrects = (perturbed_predictions == labels).float().sum().data.cpu().numpy()
			# print((t_pred != perturbed_predictions).sum())

			# label changing
			n_changed = (perturbed_predictions != original_predictions).float().sum().data.cpu().numpy()

			replace_rate = (self.args.max_steps - (final_word_ids == word_ids).sum(-1))*1.0/lengths

			n_changed_words = self.args.max_steps - (final_word_ids == word_ids).sum(-1)

			change_labels = (perturbed_predictions != original_predictions)

			replace_rate *= change_labels
			n_changed_words *= change_labels

			all_replace_rate.append(replace_rate)
			all_n_change_words.append(n_changed_words)

			# if True:
			if self.n_samples_to_disk > 0:
				# TODO: write samples to disk
				self.write_to_disk(final_word_ids=final_word_ids, perturbed_predictions=perturbed_predictions, original_predictions=original_predictions,
								   intermediate_distances=intermediate_distances, intermediate_word_ids=intermediate_word_ids,
								   sample_ids=sample_ids, lengths=lengths, intermediate_pred_probs=intermediate_pred_probs)

			results['original_corrects'] += original_corrects
			results['perturbed_corrects:'] += perturbed_corrects
			results['n_changed'] += n_changed
			results['n_samples'] += cur_batch_size
			if self.n_samples_to_disk < 0 and self.args.n_samples_to_disk > 0:
				print('Writing examples finished')
				break

		all_replace_rate = torch.cat(all_replace_rate, dim=0)
		all_n_change_words = torch.cat(all_n_change_words, dim=0)

		all_replace_rate = all_replace_rate.data.cpu().numpy()
		all_n_change_words = all_n_change_words.data.cpu().numpy()

		all_replace_rate_success = []
		all_n_change_words_success = []

		for idx, item in enumerate(all_n_change_words):
			if item > 1e-6:
				all_replace_rate_success.append(all_replace_rate[idx])
				all_n_change_words_success.append(item)
		all_replace_rate_success = np.asarray(all_replace_rate_success)
		all_n_change_words_success = np.asarray(all_n_change_words_success)

		# print and write results
		print('original_acc = {}'.format(results['original_corrects']*1.0/results['n_samples']))
		print('acc_perturbed = {}'.format(results['perturbed_corrects:']*1.0/results['n_samples']))
		print('changed_rate = {}'.format(results['n_changed']*1.0/results['n_perturbed']))

		print('replace_rate_avg = {}'.format(np.mean(all_replace_rate_success)))
		print('replace_rate_max = {}'.format(np.max(all_replace_rate_success)))
		print('replace_rate_min = {}'.format(np.min(all_replace_rate_success)))
		print('replace_rate_median = {}'.format(np.median(all_replace_rate_success)))
		print('replace_rate_var = {}'.format(np.var(all_replace_rate_success)))

		print('n_changed_words_avg = {}'.format(np.mean(all_n_change_words_success)))
		print('n_changed_words_max = {}'.format(np.max(all_n_change_words_success)))
		print('n_changed_words_min = {}'.format(np.min(all_n_change_words_success)))
		print('n_changed_words_median = {}'.format(np.median(all_n_change_words_success)))
		print('n_changed_words_var = {}'.format(np.var(all_n_change_words_success)))

		print('n_perturbed = {}'.format(results['n_perturbed']))
		sys.stdout.flush()

		self.out.write('original_acc = {}\n'.format(results['original_corrects']*1.0/results['n_samples']))
		self.out.write('acc_perturbed = {}\n'.format(results['perturbed_corrects:']*1.0/results['n_samples']))
		self.out.write('changed_rate = {}\n'.format(results['n_changed']*1.0/results['n_perturbed']))

		self.out.write('replace_rate_avg = {}\n'.format(np.mean(all_replace_rate_success)))
		self.out.write('replace_rate_max = {}\n'.format(np.max(all_replace_rate_success)))
		self.out.write('replace_rate_min = {}\n'.format(np.min(all_replace_rate_success)))
		self.out.write('replace_rate_median = {}\n'.format(np.median(all_replace_rate_success)))
		self.out.write('replace_rate_var = {}\n'.format(np.var(all_replace_rate_success)))

		self.out.write('n_changed_words_avg = {}\n'.format(np.mean(all_n_change_words_success)))
		self.out.write('n_changed_words_max = {}\n'.format(np.max(all_n_change_words_success)))
		self.out.write('n_changed_words_min = {}\n'.format(np.min(all_n_change_words_success)))
		self.out.write('n_changed_words_median = {}\n'.format(np.median(all_n_change_words_success)))
		self.out.write('n_changed_words_var = {}\n'.format(np.var(all_n_change_words_success)))

		self.out.write('n_perturbed = {}\n'.format(results['n_perturbed']))
		self.out.flush()

	def write_to_disk(self, final_word_ids, perturbed_predictions, original_predictions,
								   intermediate_distances, intermediate_word_ids, sample_ids, lengths, intermediate_pred_probs):
		"""

		:param final_word_ids: [batch_size, max_steps]
		:param perturbed_predictions: [batch_size]
		:param original_predictions: [batch_size]
		:param intermediate_distances: [batch_size, n_loops+1]
		:param intermediate_word_ids: [batch_size, n_loops+1, max_steps]
		:param sample_ids: [batch_size]
		:return:
		"""
		# [batch_size]
		# only record examples which have successfully been changed
		change_labels = (perturbed_predictions != original_predictions)
		cur_batch_size = final_word_ids.size(0)

		for idx in range(cur_batch_size):
			if not change_labels[idx]:
				continue
			with open(os.path.join(self.summary_dir, '{}.md'.format(sample_ids[idx])), 'w') as file:
				msg, change_indices, replace_dic = self.write_sample(final_word_ids=final_word_ids[idx], perturbed_prediction=perturbed_predictions[idx],
								  original_prediction=original_predictions[idx], intermediate_distances=intermediate_distances[idx],
								  intermediate_word_ids=intermediate_word_ids[idx], sample_id=sample_ids[idx], length=lengths[idx],
										intermediate_pred_probs=intermediate_pred_probs[idx])
				file.write('sample id = {}\n'.format(sample_ids[idx]))
				file.write('original prediction = {}, perturbed prediction = {}\n'.format(original_predictions[idx], perturbed_predictions[idx]))
				print('sample id = {}'.format(sample_ids[idx]))
				print('original prediction = {}, perturbed prediction = {}'.format(original_predictions[idx], perturbed_predictions[idx]))
				for m in msg:
					file.write(m+'\n')
					print(m)
				file.write('---------------\n')
				print('---------------')

				words = self.samples[sample_ids[idx]].sentence
				length = lengths[idx]
				out = ''
				for i, w in enumerate(words):
					if i == length:
						break
					if i != 0 and i % 25 == 0:
						file.write('\n')
						out += '\n'
					if i in change_indices:
						file.write('**'+w + '({})'.format(replace_dic[w]) +'**' + ' ')
						out += '**'+w+'**' + ' '
					else:
						file.write(w + ' ')
						out += w + ' '
				print(out)
				self.n_samples_to_disk -= 1
				file.flush()
				sys.stdout.flush()

	def write_sample(self, final_word_ids, perturbed_prediction, original_prediction,
					 intermediate_distances, intermediate_word_ids, sample_id, length, intermediate_pred_probs):
		"""
		write a single example to disk
		:param final_word_ids: [max_steps]
		:param perturbed_prediction: scalar
		:param original_prediction: scalar
		:param intermediate_distances: [n_loops+1]
		:param intermediate_word_ids: [n_loops+1, max_steps]
		:param sample_id: scalar
		:return:
		"""
		n_loops = intermediate_distances.size(0)

		original_word_ids = intermediate_word_ids[0]
		original_distance = intermediate_distances[0]

		# init previous distance
		previous_word_ids = intermediate_word_ids[0]
		previous_distance = intermediate_distances[0]
		previous_prob = intermediate_pred_probs[0]

		msg = []
		change_indices = []
		replace_dic = {}
		for idx in range(1, n_loops):
			# compare each set of word_ids with word_ids in the previous loop, skip if no changes
			cur_word_ids = intermediate_word_ids[idx]
			cur_distance = intermediate_distances[idx]
			cur_prob = intermediate_pred_probs[idx]

			if (cur_word_ids == previous_word_ids).sum() == self.args.max_steps:
				# no change, skip
				# assert previous_distance == cur_distance
				continue

			# change, but check only one word has been changed
			assert (cur_word_ids != previous_word_ids).sum() == 1

			# check which word has been changed
			changed_word_idx = torch.argmin((cur_word_ids == previous_word_ids).float())

			previous_word_id = int(previous_word_ids[changed_word_idx].data.cpu().numpy())
			new_word_id = int(cur_word_ids[changed_word_idx].data.cpu().numpy())

			previous_word = self.text_data.id2word[previous_word_id]
			new_word = self.text_data.id2word[new_word_id]
			replace_dic[previous_word] = new_word
			cur_msg = '{}({})_{:0.4f}_{:0.4f} -> {}({})_{:0.4f}_{:0.4f} ({}/{})'.format(previous_word, previous_word_id,
														 previous_distance, previous_prob, new_word, new_word_id,
																cur_distance, cur_prob, changed_word_idx, length)
			change_indices.append(changed_word_idx)
			msg.append(cur_msg)
			# update previous_word_ids and previous_distance
			previous_word_ids = cur_word_ids
			previous_distance = cur_distance
			previous_prob = cur_prob

		return msg, change_indices, replace_dic

