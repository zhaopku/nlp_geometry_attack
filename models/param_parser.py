import argparse

class ParamParser:
	def __init__(self):
		self.args = None

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		# dataset options
		data_args = parser.add_argument_group('Dataset options')
		data_args.add_argument('--data_dir', type=str, default='./dataset')
		data_args.add_argument('--dataset', type=str, default='imdb')
		data_args.add_argument('--train_path', type=str, default='train')
		data_args.add_argument('--test_path', type=str, default='test')
		data_args.add_argument('--val_path', type=str, default='val')

		data_args.add_argument('--vocab_size', type=int, default=60000, help='max vocab size, -1 indicates unlimited')
		data_args.add_argument('--max_length', type=int, default=600, help='max length for each example')

		data_args.add_argument('--embedding_file', type=str, default='dataset/glove.840B.300d.txt', help='pretrained word embeddings')

		data_args.add_argument('--summary_dir', type=str, default='summaries')
		data_args.add_argument('--result_dir', type=str, default='result')
		data_args.add_argument('--model_dir', type=str, default='saved_models')
		data_args.add_argument('--num_worker', type=int, default=0, help='number of worker for loading samples')
		data_args.add_argument('--abandon_stopwords', action='store_true', help='abandon stopwords while attacking')

		data_args.add_argument('--sentiment_path', type=str, default='./dataset/opinion-lexicon/sentiment-words.txt')

		# nn options
		nn_args = parser.add_argument_group('Model options')
		nn_args.add_argument('--embedding_size', type=int, default=300)
		nn_args.add_argument('--hidden_size', type=int, default=200)
		nn_args.add_argument('--model', type=str, default='lstm', help='sentence encoder')
		nn_args.add_argument('--embedding', type=str, default='glove.6B')
		nn_args.add_argument('--freeze_embedding', action='store_true', help='whether or not to freeze embedding, do not update by default')
		nn_args.add_argument('--max_steps', type=int, default=300, help='number of max steps in RNN')
		nn_args.add_argument('--n_classes', type=int, default=2)
		nn_args.add_argument('--bidirectional', action='store_true', help='whether or not to use bidirectional lstm')
		nn_args.add_argument('--mlp_size', type=int, default=150, help='hidden_size for MLP')
		# for conv
		nn_args.add_argument('--kernel_size', type=int, default=3)

		# training options
		training_args = parser.add_argument_group('Training options')
		training_args.add_argument('--batch_size', type=int, default=200)
		training_args.add_argument('--epochs', type=int, default=200, help='number of training epochs')
		training_args.add_argument('--load_model', action='store_true', help='whether or not to use old models')
		training_args.add_argument('--model_path', type=str, help='where to load model, only useful when load_model is True')
		training_args.add_argument('--robust_model_path', type=str)
		training_args.add_argument('--lr', type=float, default=1e-3)
		training_args.add_argument('--splits', type=int, default=1500, help='split into pieces')

		training_args.add_argument('--resume', type=int, default=-1, help='resume from a previous epoch. -1: no resuming; 999: latest epoch')

		# attack options
		attack_args = parser.add_argument_group('Attack options')
		attack_args.add_argument('--attack', type=str, help='what attack to use, None means no attacking')
		attack_args.add_argument('--max_loops', type=int, default=50, help='max number of loops for an attack')
		attack_args.add_argument('--metric', type=str, default='projection', help='could also be distance')
		attack_args.add_argument('--perturb_correct', action='store_true')

		attack_args.add_argument('--n_samples_to_disk', type=int, default=100)

		attack_args.add_argument('--adv', action='store_true', help='start adv training if true')
		attack_args.add_argument('--cross_section', type=str, default=None, choices=['deepfool-random'])
		attack_args.add_argument('--verbose', type=str, default='imdb-cnn-clean')

		# for generic attack

		# analyzing options
		analyzing_args = parser.add_argument_group('Analyzing options')
		analyzing_args.add_argument('--analyze', action='store_true', help='analyze sentence embeddings')
		analyzing_args.add_argument('--cos_analyze', action='store_true', help='analyze cosine values')

		# search sample id
		amt_args = parser.add_argument_group('AMT options')
		amt_args.add_argument('--search_id', action='store_true', help='search sample id of pwws examples')
		amt_args.add_argument('--pwws_path', default='amt_evaluation', help='file path for pwws')

		return parser.parse_args(args)
