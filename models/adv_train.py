import torch
import numpy
from models.greedy_attack import GreedyAttack
from copy import deepcopy
from tqdm import tqdm
import sys
import numpy as np
import os
import math

class AdvTrain:
	def __init__(self, args, text_data, model_dir, summary_dir, out, model, optimizer, loss, writer,
				 global_train_step, global_val_step, global_test_step, train_loader, val_loader, test_loader, new_epoch):
		self.args = args
		self.text_data = text_data
		self.model_dir = model_dir
		self.summary_dir = summary_dir
		self.out = out
		self.model = model

		self.optimizer = optimizer
		self.loss = loss
		self.writer = writer

		self.global_train_step = 0
		self.global_val_step = 0
		self.global_test_step = 0
		self.new_epoch = new_epoch

		if self.args.resume > 0:
			self.global_train_step = global_train_step
			self.global_val_step = global_val_step
			self.global_test_step = global_test_step

		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.greedy_attack = GreedyAttack(args=self.args, text_data=self.text_data, writer=self.writer, summary_dir=self.summary_dir, out=self.out)

	def main_loops(self):
		if self.args.resume < 0:
			self.global_test_step -= math.ceil(len(self.test_loader.dataset) * 1.0 / self.args.batch_size)
			self.train(e=-1, mode='test')

		if self.args.resume >= 0:
			print('From epoch {}'.format(self.new_epoch))
			print('train step = {}, val step = {}, test step = {}'.format(self.global_train_step, self.global_val_step, self.global_test_step))
			self.out.write('From epoch {}\n'.format(self.new_epoch))
			self.out.write('train step = {}, val step = {}, test step = {}\n'.format(self.global_train_step, self.global_val_step, self.global_test_step))

		for e in range(self.args.epochs):
			if self.args.resume > 0:
				e += self.new_epoch
			print('============ Epoch {} ============'.format(e))
			self.out.write('============ Epoch {} ============\n'.format(e))
			self.out.flush()
			sys.stdout.flush()

			self.train(e=e, mode='train')
			# self.train(e=e, mode='val')
			self.train(e=e, mode='test')

			# only save model after a full epoch
			torch.save((self.model.state_dict(), self.optimizer.state_dict(), self.global_train_step, self.global_val_step,
						self.global_test_step),
					   os.path.join(self.model_dir, str(e) + '.pth'))

	def train(self, e, mode='train'):
		if torch.cuda.is_available():
			self.model.cuda()

		if mode == 'train':
			loader = self.train_loader
			step = self.global_train_step
		elif mode == 'val':
			loader = self.val_loader
			step = self.global_val_step
		else:
			loader = self.test_loader
			step = self.global_test_step

		results = {'original_acc': 0.0, 'acc_perturbed': 0.0, 'change_rate': 0.0, 'n_samples': 0,
				   'original_corrects': 0, 'perturbed_corrects:': 0, 'n_changed': 0, 'n_perturbed': 0, 'new_loss': 0.0,
					'new_corrects': 0}
		all_replace_rate = []
		all_n_change_words = []

		for idx, (sample_ids, word_ids, lengths, labels, stopwords_mask, mask) in enumerate(tqdm(loader)):
			# if idx == 5:
			# 	break
			cur_batch_size = lengths.size(0)
			if torch.cuda.is_available():
				word_ids = word_ids.cuda()
				lengths = lengths.cuda()
				labels = labels.cuda()
				stopwords_mask = stopwords_mask.cuda()
				sample_ids = sample_ids.cuda()
				mask = mask.cuda()

			# change to eval mode when perturbing, only perturb correctly classified samples
			self.model.eval()
			logits, _ = self.model(word_ids, lengths)
			predictions = torch.argmax(logits, -1)
			original_corrects = (predictions == labels).float().sum().data.cpu().numpy()
			original_wrong = cur_batch_size - original_corrects
			correct_mask = (predictions == labels)
			wrong_mask = ~correct_mask

			c_word_ids = torch.masked_select(word_ids, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
			c_lengths = torch.masked_select(lengths, correct_mask)
			c_labels = torch.masked_select(labels, correct_mask)
			c_stopwords_mask = torch.masked_select(stopwords_mask, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
			c_mask = torch.masked_select(mask, correct_mask.view(-1, 1)).view(int(original_corrects), -1)
			c_sample_ids = torch.masked_select(sample_ids, correct_mask)

			if original_wrong > 0:
				w_word_ids = torch.masked_select(word_ids, wrong_mask.view(-1, 1)).view(int(original_wrong), -1)
				w_lengths = torch.masked_select(lengths, wrong_mask)
				w_labels = torch.masked_select(labels, wrong_mask)
				w_stopwords_mask = torch.masked_select(stopwords_mask, wrong_mask.view(-1, 1)).view(int(original_wrong), -1)
				w_mask = torch.masked_select(mask, wrong_mask.view(-1, 1)).view(int(original_wrong), -1)
				w_sample_ids = torch.masked_select(sample_ids, wrong_mask)

			cur_n_perturbed = c_mask.size(0)

			self.model.zero_grad()
			final_r_tot, final_word_ids, perturbed_predictions, intermediate_normals, intermediate_cosines,\
			intermediate_distances, original_predictions, intermediate_word_ids, intermediate_pred_probs = \
				self.greedy_attack.adv_attack(word_ids=c_word_ids, lengths=c_lengths, labels=c_labels,
											  sample_ids=c_sample_ids, model=self.model, samples=self.text_data.train_samples,
											  stopwords_mask=c_stopwords_mask, mask=c_mask)

			perturbed_corrects = (perturbed_predictions == c_labels).float().sum().data.cpu().numpy()
			# print((t_pred != perturbed_predictions).sum())

			# label changing
			n_changed = (perturbed_predictions != original_predictions).float().sum().data.cpu().numpy()
			replace_rate = (self.args.max_steps - (final_word_ids == c_word_ids).sum(-1))*1.0/c_lengths
			n_changed_words = self.args.max_steps - (final_word_ids == c_word_ids).sum(-1)
			change_labels = (perturbed_predictions != original_predictions)

			replace_rate *= change_labels
			n_changed_words *= change_labels

			all_replace_rate.append(replace_rate)
			all_n_change_words.append(n_changed_words)

			# train model
			# switch to train mode
			if mode == 'train':
				self.model.train()

			# for final_word_ids, only feed examples that successfully fool the network
			if n_changed != 0:
				final_word_ids_fooled = torch.masked_select(final_word_ids, mask=change_labels.view(-1, 1)).view(-1, self.args.max_steps)
				final_labels_fooled = torch.masked_select(c_labels, mask=change_labels)
				final_lengths_fooled = torch.masked_select(c_lengths, mask=change_labels)

				# augment with adv examples
				original_word_ids = deepcopy(word_ids)
				original_labels = deepcopy(labels)
				original_lengths = deepcopy(lengths)

				new_word_ids = torch.cat([final_word_ids_fooled, original_word_ids], dim=0)
				new_labels = torch.cat([final_labels_fooled, original_labels], dim=0)
				new_lengths = torch.cat([final_lengths_fooled, original_lengths], dim=0)
			else:
				# augment with adv examples
				original_word_ids = deepcopy(word_ids)
				original_labels = deepcopy(labels)
				original_lengths = deepcopy(lengths)

				new_word_ids = deepcopy(word_ids)
				new_labels = deepcopy(labels)
				new_lengths = deepcopy(lengths)

			# if original_wrong > 0:
			# 	new_word_ids = torch.cat([final_word_ids, w_word_ids], dim=0)
			# 	new_labels = torch.cat([c_labels, w_labels], dim=0)
			# 	new_lengths = torch.cat([c_lengths, w_lengths], dim=0)
			# else:
			# 	new_word_ids = final_word_ids
			# 	new_labels = c_labels
			# 	new_lengths = c_lengths

			new_logits, _ = self.model(new_word_ids, new_lengths)
			new_predictions = torch.argmax(new_logits, dim=-1)
			new_corrects = (new_predictions == new_labels).sum()

			new_loss = self.loss(new_logits, new_labels)
			# print('new_corrects = {}, n_changed = {}, original_correct = {}'.format(new_corrects, n_changed, original_corrects))
			# self.model.train()
			# new_logits2, _ = self.model(new_word_ids, new_lengths)
			# new_predictions2 = torch.argmax(new_logits2, dim=-1)
			# new_corrects2 = (new_predictions2 == new_labels).sum()
			# new_loss2 = self.loss(new_logits, new_labels)

			# update if in train mode
			if mode == 'train':
				self.model.zero_grad()
				new_loss.mean().backward()
				self.optimizer.step()

			# record
			self.writer.add_scalar('{}/new_loss_avg_batch'.format(mode), new_loss.sum().cpu().data * 1.0 / cur_batch_size, step)

			self.writer.add_scalar('{}/n_samples_batch'.format(mode), cur_batch_size, step)
			self.writer.add_scalar('{}/new_corrects_batch'.format(mode), new_corrects, step)
			self.writer.add_scalar('{}/new_acc_batch'.format(mode), new_corrects * 1.0 / cur_batch_size, step)

			self.writer.add_scalar('{}/n_perturbed_batch'.format(mode), cur_n_perturbed, step)
			self.writer.add_scalar('{}/perturbed_corrects_batch'.format(mode), perturbed_corrects, step)
			self.writer.add_scalar('{}/perturbed_acc_batch'.format(mode), perturbed_corrects*1.0/cur_n_perturbed*1.0, step)

			self.writer.add_scalar('{}/original_corrects_batch'.format(mode), original_corrects, step)
			self.writer.add_scalar('{}/original_acc_batch'.format(mode), original_corrects*1.0/cur_batch_size*1.0, step)

			self.writer.add_scalar('{}/n_changed_batch'.format(mode), n_changed, step)
			self.writer.add_scalar('{}/changed_rate_batch'.format(mode), n_changed*1.0/cur_n_perturbed*1.0, step)

			# replace_rate and n_changed_words of current batch
			# only record successful adv examples
			if n_changed != 0:
				replace_rate = deepcopy(replace_rate.data)
				n_changed_words = deepcopy(n_changed_words.data)
				replace_rate = torch.masked_select(replace_rate, mask=change_labels)
				n_changed_words = torch.masked_select(n_changed_words, mask=change_labels)

				replace_rate = replace_rate.data.cpu().numpy()
				n_changed_words = n_changed_words.data.cpu().numpy()
			else:
				replace_rate = np.zeros(5)
				n_changed_words = np.zeros(5)

			self.writer.add_scalar('{}/replace_rate_avg_batch'.format(mode), np.mean(replace_rate), step)
			self.writer.add_scalar('{}/replace_rate_median_batch'.format(mode), np.median(replace_rate), step)
			self.writer.add_scalar('{}/replace_rate_max_batch'.format(mode), np.max(replace_rate), step)
			self.writer.add_scalar('{}/replace_rate_min_batch'.format(mode), np.min(replace_rate), step)
			self.writer.add_scalar('{}/replace_rate_var_batch'.format(mode), np.var(replace_rate), step)

			self.writer.add_scalar('{}/n_changed_words_avg_batch'.format(mode), np.mean(n_changed_words), step)
			self.writer.add_scalar('{}/n_changed_words_median_batch'.format(mode), np.median(n_changed_words), step)
			self.writer.add_scalar('{}/n_changed_words_max_batch'.format(mode), np.max(n_changed_words), step)
			self.writer.add_scalar('{}/n_changed_words_min_batch'.format(mode), np.min(n_changed_words), step)
			self.writer.add_scalar('{}/n_changed_words_var_batch'.format(mode), np.var(n_changed_words), step)

			results['n_perturbed'] += cur_n_perturbed
			results['original_corrects'] += original_corrects
			results['perturbed_corrects:'] += perturbed_corrects
			results['n_changed'] += n_changed
			results['n_samples'] += cur_batch_size
			results['new_loss'] += new_loss.data.sum()

			step += 1

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
		print('------ {} ------'.format(mode))
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

		self.out.write('------ {} ------\n'.format(mode))
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

		self.writer.add_scalar('{}/original_acc'.format(mode), results['original_corrects']*1.0/results['n_samples'], e)
		self.writer.add_scalar('{}/acc_perturbed'.format(mode), results['perturbed_corrects:']*1.0/results['n_samples'], e)
		self.writer.add_scalar('{}/changed_rate'.format(mode), results['n_changed']*1.0/results['n_perturbed'], e)

		self.writer.add_scalar('{}/replace_rate_avg'.format(mode), np.mean(all_replace_rate_success), e)
		self.writer.add_scalar('{}/replace_rate_max'.format(mode), np.max(all_replace_rate_success), e)
		self.writer.add_scalar('{}/replace_rate_min'.format(mode), np.min(all_replace_rate_success), e)
		self.writer.add_scalar('{}/replace_rate_median'.format(mode), np.median(all_replace_rate_success), e)
		self.writer.add_scalar('{}/replace_rate_var'.format(mode), np.var(all_replace_rate_success), e)

		self.writer.add_scalar('{}/n_changed_words_avg'.format(mode), np.mean(all_n_change_words_success), e)
		self.writer.add_scalar('{}/n_changed_words_max'.format(mode), np.max(all_n_change_words_success), e)
		self.writer.add_scalar('{}/n_changed_words_min'.format(mode), np.min(all_n_change_words_success), e)
		self.writer.add_scalar('{}/n_changed_words_median'.format(mode), np.median(all_n_change_words_success), e)
		self.writer.add_scalar('{}/n_changed_words_var'.format(mode), np.var(all_n_change_words_success), e)
		self.writer.add_scalar('{}/n_perturbed'.format(mode), results['n_perturbed'], e)
		self.writer.flush()

		# restore step
		if mode == 'val':
			self.global_val_step = step
		elif mode == 'train':
			self.global_train_step = step
		else:
			self.global_test_step = step
