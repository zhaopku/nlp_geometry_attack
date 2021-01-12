from nltk.corpus import stopwords
import string


class Sample:
	def __init__(self, data, words, steps, label, length, id):
		self.word_ids = data[0:steps]
		if words is not None:
			self.sentence = words[0:steps]
		self.length = length
		self.label = label
		self.id = id
		self.history = []
		self.new_info = None

		self.mask = None
		self.stopwords_mask = None

		self.stopwords = set(stopwords.words('english'))
		self.punctuations = string.punctuation

	def set_mask(self, mask, stopwords_mask):
		self.mask = mask
		self.stopwords_mask = stopwords_mask


	def set_new_info(self, new_info):
		# [new_id, new_word, old_id, old_word, idx]
		self.new_info = new_info

class Batch:
	def __init__(self, samples):
		self.samples = samples
		self.batch_size = len(samples)
