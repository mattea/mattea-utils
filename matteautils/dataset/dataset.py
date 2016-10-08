from collections import Counter
import numpy as np
import random
from ..parser import tokenize
import codecs
import os
from itertools import chain
from scipy.spatial import distance

# Data types - Filled in by sub modules
dtypes = {
}


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

	@classmethod
	def load(cls, path, dtype="auto"):
		if dtype == "auto":
			for dt, dcls in dtypes.items():
				if dcls.identify(path):
					dtype = dt
					break

		return dtypes[dtype](path)

	@classmethod
	def identify(cls, path, dtype="auto"):
		if dtype == "auto":
			for dt, dcls in dtypes.items():
				if dcls.identify(path):
					dtype = dt
					break

		return dtypes[dtype]

	def train(self):
		return self._train

	def test(self):
		return self._test

	def valid(self):
		return self._valid

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		return iter(chain(*iter(self.data)))

	def weight(self, aggregator=sum, metric='cosine'):
		distfn = lambda x, y: distance.cdist([x], [y], metric=metric)[0]
		for s in self:
			if "weights" in s:
				return
			vec = s["vector"]
			vsum = sum(vec)
			vlen = len(vec)
			s["weights"] = [distfn(vsum / vlen, (vsum - v) / (vlen - 1)) for v in vec]

	def add_sentence(self, sid, sent):
		if type(sent) == list:
			pos = self.add_tokens(sent)
		else:
			pos = self.add_text(sent)

		self.sids[sent] = pos

	def add_text(self, text):
		return self.add_tokens(tokenize(text))

	def add_tokens(self, words):
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
		for itm in self.data:
			for word in itm.wv_text():
				yield word

	def wv_sentences(self):
		for itm in self.data:
			for sent in itm.wv_sentences():
				yield sent

	def sentences(self):
		for itm in self.data:
			for sent in itm.sentences():
				yield sent

	def vocab(self):
		try:
			return self._vocab
		except AttributeError:
			pass
		res = Counter()
		for sent in self.sentences():
			res.update(sent)
		self._vocab = res
		return res

	def wv_vocab(self):
		try:
			return self._wv_vocab
		except AttributeError:
			pass
		res = Counter()
		res.update(self.wv_text())
		self._wv_vocab = res
		res.total = sum(res.values())
		return res

	def normalize(self, matcher, df):
		for itm in self.data:
			itm.normalize(matcher, df)

	def vectorize(self, wordvec):
		for itm in self.data:
			itm.vectorize(wordvec)

	def maxShortSentence(self):
		l = 0
		for pair in self.data:
			cl = min(len(pair.s1["wv_tokens"]), len(pair.s2["wv_tokens"]))
			if cl > l:
				l = cl
		return l

	@classmethod
	def writer(self, *args, **kwargs):
		return self._writer(*args, **kwargs)


class MatchWriter(object):

	def __init__(self, sf):
		self.sh = codecs.open(sf or os.devnull, 'w', 'utf-8')
		self.writeheader()

	def __enter__(self):
		if self.sh:
			self.sh.__enter__()
		return self

	def __exit__(self, ctype, value, traceback):
		if self.sh:
			self.sh.__exit__(ctype, value, traceback)

	def writeheader(self):
		print >>self.sh, "\t".join(("QueryID", "UpdateID", "NuggetID", "Start",
																"End", "AutoP", "Score", "Label", "Update_Text",
																"Nugget_Text"))

	def write(self, pair, match):
		qid = pair.s1["query_id"]
		nugget = pair.s1
		update = pair.s2
		print >>self.sh, "\t".join((qid, update["id"], nugget["id"],
																str(match.start), str(match.end),
																str(match.autop), "%g" % match.score,
																"%g" % pair.label,
																update["text"], nugget["text"]))


class SentencePair(object):

	def __init__(self, s1, s2, sid1=None, sid2=None, pid=None, label=None):
		self.s1 = {}
		self.s2 = {}
		if type(s1) == list:
			s1toks = s1
			s1 = " ".join(s1toks)
		elif type(s1) == dict:
			self.s1 = s1
		else:
			s1toks = tokenize(s1)

		if type(s2) == list:
			s2toks = s2
			s2 = " ".join(s2toks)
		elif type(s2) == dict:
			self.s2 = s2
		else:
			s2toks = tokenize(s2)

		if not self.s1:
			self.s1 = {"tokens": s1toks, "text": s1}
			self.s2 = {"tokens": s2toks, "text": s2}
			if sid1 is None and pid is not None:
				sid1 = pid + "_1"
			if sid2 is None and pid is not None:
				sid2 = pid + "_2"
		else:
			sid1 = s1["id"]
			sid2 = s2["id"]
			pid = sid1 + "_" + sid2
			s1toks = s1["tokens"]
			s2toks = s2["tokens"]
		self.sid1 = sid1
		self.sid2 = sid2
		self.pid = pid
		self.label = label
		self.__len = len(s1toks) + len(s2toks)

	def __str__(self):
		return " ".join(self.s1["tokens"]) + " <s1e> " + " ".join(self.s2["tokens"])

	def __iter__(self):
		for itm in (self.s1, self.s2):
			yield itm

	def wv_text(self):
		for word in chain(self.s1["wv_tokens"], self.s2["wv_tokens"]):
			yield word

	def wv_sentences(self):
		return [self.s1["wv_tokens"], self.s2["wv_tokens"]]

	def sentences(self):
		return [self.s1["tokens"], self.s2["tokens"]]

	def __getitem__(self, index):
		s1l = len(self.s1["tokens"])
		if s1l > index:
			return self.s1["tokens"][index]
		else:
			return self.s2["tokens"][index - s1l]

	def __setitem__(self, index, val):
		s1l = len(self.s1["tokens"])
		if s1l > index:
			self.s1["tokens"][index] = val
		else:
			self.s2["tokens"][index - s1l] = val

	def __len__(self):
		return self.__len

	def normalize(self, matcher, df):
		self.s1["vector"], self.s1["vector_sum"] = matcher.normalize(self.s1["vector"], df)
		self.s2["vector"], self.s2["vector_sum"] = matcher.normalize(self.s2["vector"], df)

	def vectorize(self, wordvec):
		self.s1["vector"], self.s1["wv_tokens"] = wordvec.get_sentvec(self.s1["tokens"])
		self.s2["vector"], self.s2["wv_tokens"] = wordvec.get_sentvec(self.s2["tokens"])
