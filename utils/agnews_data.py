import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
import nltk
from torch.utils.data.dataset import Dataset
import torch
from utils.data_utils import Sample, Batch
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet
from utils.imdb_data import MyDataSet
import csv

class AGNewsData:
	def __init__(self, args):
		self.args = args

		#note: use 20k most frequent words
		self.UNK_WORD = '<unk>'
		self.PAD_WORD = '<pad>'

		# list of batches
		self.train_batches = []
		self.val_batches = []
		self.test_batches = []

		self.word2id = {}
		self.id2word = {}

		self.train_samples = None
		self.val_samples = None
		self.test_samples = None
		self.pre_trained_embedding = None
		self.stopwords = set(stopwords.words('english'))
		self.word2synonyms = {}
		self.wordid2synonyms = {}

		self.vocab = None
		self.vocab_ids = None

		# PyTorch dataset
		self.training_set = None
		self.val_set = None
		self.test_set = None

		self.create()


	def construct_vocab(self):
		vocab = []
		vocab_ids = []
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]
			if word == self.PAD_WORD or word == self.UNK_WORD:
				continue
			else:
				vocab.append(word)
				vocab_ids.append(word_id)

		return vocab, vocab_ids

	def construct_synonyms(self):
		"""
		for each word in the vocab, find its synonyms
		build a dictionary, where key is word, value is its synonyms
		:return:
		"""
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]

			if word == self.PAD_WORD or word == self.UNK_WORD:
				self.word2synonyms[word] = [word]
				self.wordid2synonyms[word_id] = [word_id]
				continue

			synonyms = []
			synonyms_id = []

			for syn in wordnet.synsets(word):
				for l in syn.lemmas():
					w = l.name()
					if w not in self.word2id.keys():
						continue
					w_id = self.word2id[w]
					# if synonym is PAD or UNK, continue
					if w_id == self.word2id[self.PAD_WORD] or w_id == self.word2id[self.UNK_WORD]:
						continue
					synonyms.append(w)
					synonyms_id.append(w_id)
			# put original word in synonyms
			synonyms.append(word)
			synonyms_id.append(word_id)
			synonyms = list(set(synonyms))
			synonyms_id = list(set(synonyms_id))

			self.word2synonyms[word] = synonyms
			self.wordid2synonyms[word_id] = synonyms_id

	def create(self):
		self.train_samples, self.val_samples, self.test_samples = self._create_data()

		# [num_batch, batch_size, max_step]
		self.train_batches = self._create_batch(self.train_samples, tag='train')
		self.val_batches = self._create_batch(self.val_samples)
		self.test_batches = self._create_batch(self.test_samples)

		print('Dataset created')

	def construct_dataset(self, max_steps):
		self.training_set = MyDataSet(samples=self.train_samples, max_steps=max_steps)
		self.val_set = MyDataSet(samples=self.val_samples, max_steps=max_steps)
		self.test_set = MyDataSet(samples=self.test_samples, max_steps=max_steps)
		self.training_set_all = MyDataSet(samples=self.train_samples+self.val_samples, max_steps=max_steps)

	def get_vocabulary_size(self):
		assert len(self.word2id) == len(self.id2word)
		return len(self.word2id)

	def _create_batch(self, all_samples, tag='test'):
		all_batches = []
		# if tag == 'train':
		# 	random.shuffle(all_samples)
		if all_samples is None:
			return all_batches

		num_batch = len(all_samples)//self.args.batch_size + 1
		for i in range(num_batch):
			samples = all_samples[i*self.args.batch_size:(i+1)*self.args.batch_size]

			if len(samples) == 0:
				continue

			batch = Batch(samples)
			all_batches.append(batch)

		return all_batches

	def _create_samples(self, path):
		oov_cnt = 0
		cnt = 0
		sample_cnt = 0
		all_samples = []

		with open(path, 'r') as file:
			reader = csv.reader(file)

			for idx, row in enumerate(tqdm(reader)):
				line = ' '.join(row[1:])
				label = int(row[0]) - 1
				words = nltk.word_tokenize(line)
				word_ids = []

				words = words[:self.args.max_length]
				if self.args.embedding == 'glove.6B':
					words = [w.lower() for w in words]
				length = len(words)
				cnt += length
				for word in words:
					if word in self.word2id.keys():
						id_ = self.word2id[word]
					else:
						id_ = self.word2id[self.UNK_WORD]
						if self.args.vocab_size == -1:
							print('Check!')
					if id_ == self.word2id[self.UNK_WORD] and word != self.UNK_WORD:
						oov_cnt += 1
					word_ids.append(id_)
				while len(word_ids) < self.args.max_length:
					word_ids.append(self.word2id[self.PAD_WORD])
				while len(words) < self.args.max_length:
					words.append(self.PAD_WORD)
				# if self.args.dataset == 'yahoo':
				# 	# yahoo dataset is too large
				# 	words = None
				sample = Sample(data=word_ids, words=words,
								steps=self.args.max_length, label=label, length=length, id=-1)
				sample_cnt += 1
				all_samples.append(sample)

		return all_samples, oov_cnt, cnt

	def create_embeddings(self):
		words = self.word2id.keys()

		glove_embed = {}

		with open(self.args.embedding_file, 'r') as glove:
			lines = glove.readlines()
			for line in tqdm(lines, desc='glove'):
				splits = line.split()
				word = splits[0]
				if len(splits) > self.args.embedding_size+1:
					word = ''.join(splits[0:len(splits) - self.args.embedding_size])
					splits[1:] = splits[len(splits) - self.args.embedding_size:]
				if word not in words:
					continue
				embed = [float(s) for s in splits[1:]]
				glove_embed[word] = embed

		embeds = []
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]
			if word in glove_embed.keys():
				embed = glove_embed[word]
			elif word == self.PAD_WORD:
				# PAD gets zeros
				embed = np.zeros(self.args.embedding_size)
			else:
				# now UNK gets all zeros
				embed = np.zeros(self.args.embedding_size)
				# embed = glove_embed[self.UNK_WORD]
				self.word2id[word] = self.word2id[self.UNK_WORD]
			embeds.append(embed)

		embeds = np.asarray(embeds)

		return embeds

	def _create_data(self):

		train_path = os.path.join(self.args.data_dir, self.args.dataset, self.args.train_path+'.csv')
		test_path = os.path.join(self.args.data_dir, self.args.dataset, self.args.test_path+'.csv')

		print('Building vocabularies for {} dataset'.format(self.args.dataset))
		self.word2id, self.id2word = self._build_vocab(train_path, test_path)

		print('Creating pretrained embeddings!')
		self.pre_trained_embedding = self.create_embeddings()

		print('Building training samples!')
		all_train_samples, train_oov, train_cnt = self._create_samples(train_path)
		# shuffle training samples here!
		random.shuffle(all_train_samples)

		n_train_samples = len(all_train_samples)

		n_val_samples = int(n_train_samples*0.2)
		n_train_samples = n_train_samples - n_val_samples
		train_samples = all_train_samples[:n_train_samples]
		val_samples = all_train_samples[n_train_samples:]

		test_samples, test_oov, test_cnt = self._create_samples(test_path)

		random.shuffle(test_samples)

		train_samples = self.create_id(train_samples, 'train')
		val_samples = self.create_id(val_samples, 'val')
		test_samples = self.create_id(test_samples, 'test')

		self.vocab, self.vocab_ids = self.construct_vocab()
		self.construct_synonyms()

		self.create_mask(train_samples, 'train')
		self.create_mask(val_samples, 'val')
		self.create_mask(test_samples, 'test')

		print('OOV rate for train&val = {:.2%}'.format(train_oov*1.0/train_cnt))
		print('OOV rate for test = {:.2%}'.format(test_oov*1.0/test_cnt))

		return train_samples, val_samples, test_samples

	def create_mask(self, samples, tag='train'):
		for sample in tqdm(samples, desc='creating mask for {}'.format(tag)):
			mask = []
			stopwords_mask = []
			for idx, word in enumerate(sample.sentence):
				if idx >= sample.length:
					mask.append(0)
					stopwords_mask.append(0)
					continue

				if word in string.punctuation or word not in self.word2id.keys() or word == self.PAD_WORD\
						or word == self.UNK_WORD or word not in self.vocab:
					mask.append(0)
				elif len(self.word2synonyms[word]) <= 1:
					mask.append(0)

				else:
					mask.append(1)

				if word.lower() in self.stopwords:
					stopwords_mask.append(0)
				else:
					stopwords_mask.append(1)

				sample.set_mask(mask=mask, stopwords_mask=stopwords_mask)

	def create_id(self, samples, tag):
		for idx, sample in enumerate(tqdm(samples, desc='creating id for {} samples'.format(tag))):
			sample.id = idx

		return samples

	def _read_sents(self, path):

		all_words = []
		with open(path, 'r') as file:
			reader = csv.reader(file)

			for idx, row in enumerate(tqdm(reader)):
				line = ' '.join(row[1:])
				words = nltk.word_tokenize(line)

				if self.args.embedding == 'glove.6B':
					words = [w.lower() for w in words]

				all_words.extend(words)

		return all_words

	def _build_vocab(self, train_path, test_path):

		all_train_words = self._read_sents(train_path)
		all_test_words = self._read_sents(test_path)

		all_words = all_train_words + all_test_words

		print('Number of unique words = ', len(list(set(all_words))))

		counter = Counter(all_words)

		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# keep the most frequent vocabSize words, including the special tokens
		# -1 means we have no limits on the number of words
		if self.args.vocab_size != -1:
			count_pairs = count_pairs[0:self.args.vocab_size-2]

		count_pairs.append((self.UNK_WORD, 100000))
		count_pairs.append((self.PAD_WORD, 100000))

		if self.args.vocab_size != -1:
			assert len(count_pairs) == self.args.vocab_size

		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(len(words))))

		id_to_word = {v: k for k, v in word_to_id.items()}

		return word_to_id, id_to_word

	def get_batches(self, tag='train'):
		if tag == 'train':
			return self._create_batch(self.train_samples, tag='train')
		elif tag == 'val':
			return self.val_batches
		else:
			return self.test_batches
