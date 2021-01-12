import os
from models.model_lstm import ModelLSTM
from copy import deepcopy
import numpy as np
import sys
import torch
from tqdm import tqdm

def construct_dir(prefix, args, create_dataset_name=False, create_dist_name=False):

	if create_dataset_name or create_dist_name:
		file_name = ''
		file_name += prefix + '-'
		file_name += str(args.vocab_size) + '-'
		if args.embedding == 'random':
			emb = 'glove'
		else:
			emb = args.embedding
		if create_dist_name:
			file_name += emb
		else:
			file_name += emb + '-'
			file_name += str(args.max_length) + '.pkl'
		return file_name

	path = ''
	path += args.dataset
	path += '_{}_'.format(args.model)
	path += 'esize_{}_'.format(args.embedding_size)
	if args.freeze_embedding:
		t = 'f'
	else:
		t = 'u'


	path += '{}_{}'.format(args.embedding, t)
	path += '_lr_'
	path += str(args.lr)
	path += '_h_'
	path += str(args.hidden_size)
	path += '_bt_'
	path += str(args.batch_size)
	path += '_s_' + str(args.max_steps)
	# path += '_m_' + str(args.mlp_size)

	if args.attack is not None:
		if args.abandon_stopwords:
			path += '_ns'
		path += '_{}_{}_{}'.format(args.attack, args.max_loops, args.metric)
		if args.perturb_correct:
			path += '_c'
		if args.adv:
			path += '_adv'

	return os.path.join(prefix, path)

def log(writer, out, results, mode, e):
	# start logging
	results['acc_epoch'] = results['corrects'] * 1.0 / (results['n_samples'] * 1.0)
	results['avg_loss'] = results['total_loss'] * 1.0 / (results['n_samples'] * 1.0)

	result_line = ''
	if mode == 'train':
		result_line += '\n---------------------- Epoch {} ----------------------\n'.format(e)
	result_line += '\n{}, '.format(mode)
	result_line += '{} = {:.5f}, '.format('acc', results['acc_epoch'])
	result_line += '{} = {:.5f}, '.format('loss_epoch_total', results['total_loss'])
	result_line += '{} = {:.5f}'.format('loss_epoch_avg', results['avg_loss'])

	print(result_line)
	out.write(result_line + '\n')
	out.flush()

	writer.add_scalar('{}/total_samples'.format(mode), results['n_samples'], e)
	writer.add_scalar('{}/total_corrects'.format(mode), results['corrects'], e)
	writer.add_scalar('{}/acc_epoch'.format(mode), results['acc_epoch'], e)
	writer.add_scalar('{}/loss_epoch_total'.format(mode), results['total_loss'], e)
	writer.add_scalar('{}/loss_epoch_avg'.format(mode), results['avg_loss'], e)

def statistics_individual(all_samples, max_steps, tag):
	all_lengths = []

	cnt_0 = 0
	cnt_1 = 0
	for sample in all_samples:
		all_lengths.append(sample.length)
		if sample.label == 0:
			cnt_0 += 1
		else:
			cnt_1 += 1

	print('\nStatistics of {}'.format(tag))
	all_lengths = np.asarray(all_lengths) - 1
	print('{}% of positive samples'.format(cnt_1*100.0/(cnt_1+cnt_0), tag))
	print('min length = {}'.format(np.min(all_lengths)))
	print('max length = {}'.format(np.max(all_lengths)))
	print('avg length = {}'.format(np.mean(all_lengths)))
	all_lengths = np.asarray(all_lengths)
	print('{} of samples have more (or the same) words than {}'.format(np.sum(all_lengths >= max_steps-1)/len(all_lengths), max_steps-1))
	sys.stdout.flush()

def statistics(text_data, max_steps):
	statistics_individual(text_data.train_samples, max_steps, 'train')
	statistics_individual(text_data.val_samples, max_steps, 'val')
	statistics_individual(text_data.test_samples, max_steps, 'test')

def compute_dist(vocab, word2id, id2word, embedding_file):
	"""

	:param embeddings: [vocab_size, embedding_size]
	:return:
	"""
	word2embed = dict()
	with open(embedding_file, 'r') as file:
		lines = file.readlines()
		for line in lines:
			splits = line.strip().split()
			word = splits[0]
			embeddings = [float(s) for s in splits[1:]]
			if len(embeddings) != 300:
				break
			word2embed[word] = np.asarray(embeddings)

	dist_vocab = set(vocab) - set(word2embed.keys())

	embeddings = torch.tensor(embeddings).float()
	if torch.cuda.is_available():
		embeddings = embeddings.cuda()

	print('embedding size = {}'.format(embeddings.size()))
	all_dist = []
	for word_idx in tqdm(range(embeddings.size(0))):
		# [1, embedding_size]
		cur_embeddings = embeddings[word_idx].unsqueeze(0)
		# [vocab_size, embedding_size]
		dist_embeddings = cur_embeddings - embeddings
		# [vocab_size]
		dist_embeddings = torch.pow(dist_embeddings, 2).sum(-1)
		nearest_word_idx = dist_embeddings.argsort(dim=-1, descending=False)[:200]
		cur_word = id2word[word_idx]
		nearest_words = [id2word[int(i.data.cpu().numpy())] for i in nearest_word_idx]
		all_dist.append(dist_embeddings.data.cpu())
		del dist_embeddings
	del embeddings
	# [vocab_size, vocab_size]
	all_dist = torch.cat(all_dist, 0)

	return all_dist