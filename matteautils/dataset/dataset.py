import collections
import numpy as np

class Dataset(object):
	def __init__(self, vocabulary_size=50000, store_data=True):
		self.data = []
		self.labels = []
		self.counter = collections.Counter()
		self.dictionary = dict()
		self.reverse_dictionary = dict()
		self.vocabulary_size = vocabulary_size
		self.size = 0
		self.store_data = store_data

	def __len__(self):
		return len(self.data)

	def add_sentence(self, sid, sent):
		if type(sent) == list:
			pos = self.add_tokens(sent)
		else:
			pos = self.add_text(sent)

		self.sids[sent] = pos

	def add_text(self, text):
		return self.add_tokens(tokenize(text))

	def add_tokens(self, words):
		self.counter.update(words)
		self.size += len(words)

		if self.store_data:
			self.data.append(words)
			off = len(self.data) - 1
		else:
			#TODO(mattea): maybe make this more useful? If I find a use for it
			off = self.size
		return off


	def add_label(self, label):
		return self.labels.append(label)


	def build_dictionary(self):
		count = [['UNK', -1]]
		count.extend(self.counter.most_common(self.vocabulary_size - 1))
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)

		count[0][1] = self.size - sum([x[1] for x in count])
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		self.count = count
		self.dictionary = dictionary
		self.reverse_dictionary = reverse_dictionary
		return count, dictionary, reverse_dictionary

	def index_data(self):
		dictionary = self.dictionary
		for data in self.data:
			for wind in range(len(data)):
				word = data[wind]
				if word in dictionary:
					index = dictionary[word]
				else:
					index = 0  # dictionary['UNK']
				data[wind] = index
		self.data = np.array(self.data)
		self.labels = np.array(self.labels)

	def split(self, p=.8, k=None):
		if k is None:
			if type(p) != list:
				p = [p, 1-p]
		else:
			p = [1/k] * k

		dlen = len(self.data)
		shuf = self.data[:]
		shuf = random.shuffle(shuf)
		spl = []
		off = 0
		for x in p:
			spl.append(shuf[int(dlen * off):int(dlen * x)])

		self.__split = spl
		return spl





