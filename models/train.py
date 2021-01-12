import torch
import os
import numpy as np
import torch.optim as optimizer
from torch.utils.data import DataLoader
from torch import nn
import argparse
from tqdm import tqdm
from utils.imdb_data import IMDBData
from models import utils
from models.model_lstm import ModelLSTM
import pickle as p
from torch.utils.tensorboard import SummaryWriter
from models.attack_samples import AttackSamples
from utils.agnews_data import AGNewsData
from models.model_cnn import ModelCNN
from models.adv_train import AdvTrain
from models.param_parser import ParamParser
from models.utils import log, statistics
from models.utils import compute_dist

class Train:
	def __init__(self):
		self.args = None

		self.data_dir = None
		self.dataset_name = None
		self.text_data = None

		self.train_loader = None
		self.val_loader = None
		self.test_loader = None

		self.model_dir = None
		self.result_dir = None
		self.summary_dir = None
		self.out_path = None

		self.out = None
		self.optimizer = None
		self.model = None
		self.loss = None
		self.writer = None
		self.attack_loop = None
		self.cosine_analyzer = None
		self.attack_samples = None

		self.global_train_step = 0
		self.global_val_step = 0
		self.global_test_step = 0

		self.adv_train = None

		self.new_epoch = -1
		self.param_parser = ParamParser()
		self.vocab = None

	def construct_data(self):
		self.data_dir = os.path.join(self.args.data_dir, self.args.dataset)
		self.dataset_name = utils.construct_dir(prefix=self.args.dataset, args=self.args, create_dataset_name=True)
		dataset_file_name = os.path.join(self.data_dir, self.dataset_name)

		if not os.path.exists(dataset_file_name):
			if self.args.dataset == 'imdb':
				self.text_data = IMDBData(args=self.args)
			elif self.args.dataset == 'agnews':
				self.text_data = AGNewsData(args=self.args)
			else:
				print('Cannot recognize {}'.format(self.args.dataset))
				raise NotImplementedError

			with open(dataset_file_name, 'wb') as datasetFile:
				p.dump(self.text_data, datasetFile)
			print('dataset created and saved to {}, exiting ...'.format(dataset_file_name))
			exit(0)
		else:
			with open(dataset_file_name, 'rb') as datasetFile:
				self.text_data = p.load(datasetFile)
			print('dataset loaded from {}'.format(dataset_file_name))
		# construct dataset
		self.text_data.construct_dataset(max_steps=self.args.max_steps)

		self.train_loader = DataLoader(dataset=self.text_data.training_set_all, num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=True)
		self.val_loader = DataLoader(dataset=self.text_data.val_set, num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)
		self.test_loader = DataLoader(dataset=self.text_data.test_set, num_workers=self.args.num_worker, batch_size=self.args.batch_size, shuffle=False)

	def construct_dir(self):
		self.model_dir = utils.construct_dir(prefix=self.args.model_dir, args=self.args, create_dataset_name=False)
		self.summary_dir = utils.construct_dir(prefix=self.args.summary_dir, args=self.args, create_dataset_name=False)
		self.out_path = utils.construct_dir(prefix=self.args.result_dir, args=self.args, create_dataset_name=False) + '.txt'

		if not os.path.exists(self.args.result_dir):
			os.makedirs(self.args.result_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		if not os.path.exists(self.summary_dir):
			os.makedirs(self.summary_dir)

	def construct_model(self):
		if self.args.model == 'lstm':
			self.model = ModelLSTM(args=self.args, pre_trained_embedding=self.text_data.pre_trained_embedding,
								   vocab_size=self.text_data.get_vocabulary_size())
		elif self.args.model == 'cnn':
			self.model = ModelCNN(args=self.args, pre_trained_embedding=self.text_data.pre_trained_embedding,
								   vocab_size=self.text_data.get_vocabulary_size())
		else:
			print('model {} not recognized'.format(self.args.model))

		self.optimizer = optimizer.Adam(self.model.parameters(), lr=self.args.lr)
		self.loss = nn.CrossEntropyLoss(reduction='none')
		self.writer = SummaryWriter(log_dir=self.summary_dir, flush_secs=5)

	def load_model(self, model_path):
		if torch.cuda.is_available():
			(model_state_dict, optimizer_state_dict,
			 self.global_train_step, self.global_val_step, self.global_test_step) = torch.load(model_path)
		else:
			(model_state_dict, optimizer_state_dict,
			 self.global_train_step, self.global_val_step, self.global_test_step) = torch.load(model_path, map_location='cpu')

		self.model.load_state_dict(model_state_dict)

		# load optimizer state when resuming from a checkpoint
		if self.args.resume > 0:
			self.optimizer.load_state_dict(optimizer_state_dict)
			if torch.cuda.is_available():
				for state in self.optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = v.cuda()
		print('model loaded from {}'.format(model_path))

	def compute_dist(self):
		self.vocab = self.text_data.get_vocab()
		dists = compute_dist(vocab=self.vocab, word2id=self.text_data.word2id, id2word=self.text_data.id2word, embedding_file='counter-fitted-vectors.txt')
		dist_name = utils.construct_dir(prefix=self.args.dataset, args=self.args, create_dist_name=True)
		with open(dist_name, 'wb') as file:
			p.dump(dists, file)

	def main(self, args=None):
		# torch.manual_seed(0)
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False
		# np.random.seed(0)
		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))
		self.args = self.param_parser.parse_args(args=args)

		self.construct_data()
		self.construct_dir()
		self.construct_model()
		self.text_data.construct_synonyms()
		statistics(self.text_data, self.args.max_steps)

		if self.args.load_model and self.args.resume == -1:
			self.load_model(model_path=self.args.model_path)


		eval_samples = None

		file_mode = 'a' if self.args.load_model and not (self.args.adv or self.args.attack) else 'w'
		if self.args.resume >= 0:
			file_mode = 'a'
		# print(file_mode)
		with open(self.out_path, file_mode) as self.out:
			# resume from a checkpoint
			if self.args.resume > 0:
				model_path = os.path.join(self.model_dir, '{}.pth'.format(self.args.resume))
				self.new_epoch = self.args.resume + 1

				# we do not load 999 epochs
				if self.args.resume == 999:
					saved_models = os.listdir(self.model_dir)
					print('model saved in {}, available models are {}'.format(self.model_dir, saved_models))
					self.out.write('model saved in {}, available models are {}\n'.format(self.model_dir, saved_models))
					saved_models = [i for i in saved_models if i.endswith('.pth')]
					saved_epochs = sorted([int(i.split('.')[0]) for i in saved_models])
					self.new_epoch = int(saved_epochs[-1]) + 1
					model_path = os.path.join(self.model_dir, '{}.pth'.format(saved_epochs[-1]))
					print('resume from {}'.format(model_path))
					self.out.write('resume from {}\n'.format(model_path))

				# if self.args.multi:
				# 	self.multi_gpu()
				self.load_model(model_path)

			if self.args.adv:
				self.adv_train = AdvTrain(args=self.args, text_data=self.text_data, model_dir=self.model_dir,
										  summary_dir=self.summary_dir, out=self.out,
										  model=self.model, optimizer=self.optimizer, loss=self.loss,
										  writer=self.writer, global_train_step=self.global_train_step,
										  global_val_step=self.global_val_step, global_test_step=self.global_test_step,
										  train_loader=self.train_loader, val_loader=self.val_loader,
										  test_loader=self.test_loader, new_epoch=self.new_epoch)
				self.adv_train.main_loops()
				exit(0)

			if self.args.attack is not None:
				if eval_samples is not None:
					samples = eval_samples
				else:
					samples = self.text_data.test_samples

				self.attack_samples = AttackSamples(args=self.args, text_data=self.text_data, out=self.out,
													  model=self.model, writer=self.writer, summary_dir=self.summary_dir, samples=samples)
				self.attack_samples.attack_all_samples()
				print('attacker exiting')
				exit(0)

			if file_mode == 'w':
				for k, v in vars(self.args).items():
					self.out.write('{} = {}\n'.format(str(k), str(v)))
				self.out.write('\n\n')
			# training loops
			self.loop()

	def loop(self):
		if torch.cuda.is_available():
			self.model.cuda()

		if self.args.resume >= 0:
			print('From epoch {}'.format(self.new_epoch))
			print('train step = {}, val step = {}, test step = {}'.format(self.global_train_step, self.global_val_step, self.global_test_step))
			self.out.write('From epoch {}\n'.format(self.new_epoch))
			self.out.write('train step = {}, val step = {}, test step = {}\n'.format(self.global_train_step, self.global_val_step, self.global_test_step))

		for e in range(self.args.epochs):
			if self.args.resume >= 0:
				e += self.new_epoch

			self.global_train_step = self.run(e, mode='train')
			# self.global_val_step = self.run(e, mode='val')
			self.global_test_step = self.run(e, mode='test')

			# save model at the end of each epoch
			torch.save((self.model.state_dict(), self.optimizer.state_dict(), self.global_train_step, self.global_val_step, self.global_test_step),
			           os.path.join(self.model_dir, str(e)+'.pth'))

	def run(self, e, mode='val'):
		if mode == 'val':
			loader = self.val_loader
			step = self.global_val_step
		elif mode == 'test':
			loader = self.test_loader
			step = self.global_test_step
		else:
			loader = self.train_loader
			step = self.global_train_step

		if mode == 'train':
			self.model.train()
		else:
			self.model.eval()

		results = {'acc_epoch': 0.0, 'total_loss': 0.0, 'avg_loss': 0.0, 'n_samples': 0, 'corrects': 0.0}

		for idx, (sample_ids, word_ids, lengths, labels, stopwords_mask, mask) in enumerate(tqdm(loader)):
			cur_batch_size = lengths.size(0)
			if torch.cuda.is_available():
				word_ids = word_ids.cuda()
				lengths = lengths.cuda()
				labels = labels.cuda()
				stopwords_mask = stopwords_mask.cuda()
				mask = mask.cuda()

			# logits, last_relevant_outputs = self.model(word_ids, lengths)
			logits, last_relevant_outputs, embedded = self.model(word_ids, lengths, return_embedded=True)

			predictions = torch.argmax(logits, dim=-1)
			corrects = (predictions == labels).sum()
			loss = self.loss(logits, labels)

			results['n_samples'] += cur_batch_size
			results['total_loss'] += loss.sum().cpu().data
			results['corrects'] += corrects

			self.writer.add_scalar('train/n_samples', cur_batch_size, self.global_train_step)
			self.writer.add_scalar('train/loss_batch_avg', loss.sum().cpu().data*1.0/cur_batch_size, self.global_train_step)
			self.writer.add_scalar('train/acc_batch', corrects*1.0/cur_batch_size, self.global_train_step)
			self.writer.add_scalar('train/corrects_batch', corrects, self.global_train_step)

			# update model
			if mode == 'train':
				self.model.zero_grad()
				loss.mean().backward()
				self.optimizer.step()

			step += 1

		# log results
		log(writer=self.writer, results=results, out=self.out, mode=mode, e=e)

		return step