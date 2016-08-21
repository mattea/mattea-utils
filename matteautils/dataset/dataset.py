from collections import Counter, defaultdict
import numpy as np
import random
from ..parser import tokenize
import codecs
import os


class Dataset(object):
	def __init__(self, vocabulary_size=50000, store_data=True):
		self.data = []
		self.labels = []
		self.counter = Counter()
		self.dictionary = dict()
		self.reverse_dictionary = dict()
		self.vocabulary_size = vocabulary_size
		self.size = 0
		self.store_data = store_data

	def train(self):
		return self._train

	def test(self):
		return self._test

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
				p = [p, 1 - p]
		else:
			p = [1 / k] * k

		dlen = len(self.data)
		shuf = self.data[:]
		shuf = random.shuffle(shuf)
		spl = []
		off = 0
		for x in p:
			spl.append(shuf[int(dlen * off):int(dlen * x)])

		self._split = spl
		return spl

	#def text(self):
	#	for pair in self.data:
	#		for word in pair:
	#			yield word

	def wv_text(self):
		for pair in self.data:
			for word in pair.wv_text():
				yield word

	def wv_sentences(self):
		for pair in self.data:
			for sent in pair.wv_sentences():
				yield sent

	def wv_vocab(self):
		res = Counter()
		res.update(self.wv_text())
		return res

	def normalize(self, matcher, df):
		for pair in self.data:
			pair.normalize(matcher, df)

	def vectorize(self, wordvec):
		for pair in self.data:
			pair.vectorize(wordvec)


class MatchWriter(object):

	def __init__(self, sf):
		self.sh = codecs.open(sf or os.devnull, 'w', 'utf-8')

	def __enter__(self):
		if self.sh:
			self.sh.__enter__()
			self.writeheader()
		return self

	def __exit__(self, ctype, value, traceback):
		if self.sh:
			self.sh.__exit__(ctype, value, traceback)

	def writeheader(self):
		print >>self.sh, "\t".join(("QueryID", "UpdateID", "NuggetID", "Start",
																"End", "AutoP", "Score", "Update_Text",
																"Nugget_Text"))

	def write(self, qid, nugget, update, match):
		print >>self.sh, "\t".join((qid, update["id"], nugget["id"],
																str(match.start), str(match.end),
																str(match.autop), "%g" % match.score,
																update["text"], nugget["text"]))
