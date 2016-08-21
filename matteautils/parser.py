#from peak.util.imports import lazyModule
from __future__ import division
import functools
stemmer = None
from collections import Counter
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stopl = None
from gensim.models import Word2Vec
import numpy as np
import heapq
#import math
from scipy.spatial.distance import cosine
import sys


def printd(s):
	print >>sys.stderr, s


class Match(object):
	def __init__(self, score=0.0, start=0, end=0, mtype=None):
		self.score = score
		self.start = start
		self.end = end
		self.mtype = mtype


wvsource = "https://docs.google.com/uc?export=download&confirm=xpiI&id=0B7XkCwpI5KDYeFdmcVltWkhtbmM"


def tonumber(word):
	try:
		return float(word)
	except Exception:
		return None


class Word(object):
	def __init__(self, text="", vector=[]):
		self.text = text
		self.vector = vector

	@classmethod
	def create(cls, text="", vector=[]):
		# TODO(mattea): temporarily just return the vector, rather than a wrapper class, for efficiency
		return vector
		num = tonumber(text)
		if num is not None:
			return NumberWord(num)
		else:
			return cls(text, vector)

	def dist(self, wv, fn='cosine'):
		if type(wv) != Word:
			return 1

		res = cosine(self.vector, wv.vector)
		return res if not np.isnan(res) else 1
		# wordvecs are already (theoretically) l2 normalized, so these are equivalent
		#return np.dot(self.vector, wv.vector)


class NumberWord(object):
	def __init__(self, num):
		self.num = num

	def dist(self, wv, fn='cosine'):
		if type(wv) != NumberWord:
			return 1

		#return math.exp(abs(self.num - wv.num)) / math.exp(max(self.num, wv.num))
		if self.num == wv.num:
			return 0
		else:
			return abs(self.num - wv.num) / max(abs(self.num), abs(wv.num))


class WordVec(object):
	wordvecf = '/huge1/sourcedata/word2vec/google/vectors-skipgram1000-en.bin'
	wordvec = []

	def __init__(self, wf=None, sentences=None, wvout=None, size=1000):
		if sentences and wvout:
			wv = Word2Vec(size=size, window=5, min_count=1, workers=5)
			wv.build_vocab(sentences)
			wv.intersect_word2vec_format(wf or self.wordvecf, binary=True)
			wv.save_word2vec_format(wvout, binary=True)
			self.wordvec = wv
		else:
			self.wordvec = Word2Vec.load_word2vec_format(wf or self.wordvecf, binary=True)

		self.size = self.wordvec.vector_size
		self.originalsize = self.wordvec.vector_size
		self.emptyvec = Word.create("", np.array([1e-9] * self.size))
		self.__logdf = None

	def logdf(self, tf=None):
		if tf is None:
			if self.__logdf:
				return self.__logdf
			return np.zeros(self.size)

		logdf = np.zeros(self.size)
		nterms = 0

		for t, tc in tf.items():
			if t in self.wordvec:
				logdf += self.wordvec[t] * tc
				nterms += 1

		self.__logdf = np.nan_to_num(np.log2(logdf))
		printd("Found %d (%0.2f%% toks, %0.2f%% wv) terms from vocab in wordvec" % (nterms, 100 * nterms / len(tf), 100 * nterms / len(self.wordvec.vocab)))
		return self.__logdf

	def normalize(self):
		wv = self.wordvec
		#wv.syn0 = ((wv.syn0 + 1) / 2).astype(np.float32)
		wv.syn0 = np.concatenate((np.maximum(0, wv.syn0), np.maximum(0, -wv.syn0)), axis=1)
		self.size *= 2
		wv.vector_size = self.size

	def get_sentvec(self, words):
		sent_vec = []
		sent = []
		witer = iter(range(len(words)))
		for i in witer:
			wvec = self.emptyvec
			if (i + 1) < len(words):
				word = words[i] + '_' + words[i + 1]
				nwvec = self.get_wordvec(word)
				if nwvec is not self.emptyvec:
					wvec = nwvec
					sent.append(word)
					witer.next()
			if wvec is self.emptyvec:
				word = words[i]
				wvec = self.get_wordvec(word)
				if wvec is self.emptyvec:
					word = stem(word)
					wvec = self.get_wordvec(word)
			if wvec is self.emptyvec:
				printd("No vector for %s or %s:" % (words[i], word))
			sent.append(word)
			sent_vec.append(wvec)
		return sent_vec, sent

#	def get_wordvec(self, word):
#		word = '/en/' + word
#		try:
#			return Word(word, self.wordvec[word])
#		except KeyError:
#			return self.emptyvec

	def get_wordvec(self, word):
		#TODO(mattea): Uncomment if we go back to using number sim
		#num = tonumber(word)
		#if num is not None:
		#	return Word.create(num, self.emptyvec)
		#word = '/en/' + word
		try:
			return Word.create(word, self.wordvec[word])
		except KeyError:
			return self.emptyvec

	def trim(self, trimfn):
		self.wordvec.scale_vocab(trim_rule=trimfn)

	def save(self, wvfile):
		self.wordvec.save_word2vec_format(wvfile, binary=True)

	def vocab(self):
		return self.wordvec.vocab


class GloveVec(WordVec):
	wordvecf = '/huge1/sourcedata/word2vec/glove/glove.42B.300d'
	wordvec = []

	def __init__(self, wf=None, sentences=None, wvout=None, size=1000):
		if sentences and wvout:
			wv = Glove(wf)
			wv = Word2Vec(size=size, window=5, min_count=1, workers=5)
			wv.build_vocab(sentences)
			wv.intersect_word2vec_format(wf or self.wordvecf, binary=True)
			wv.save_word2vec_format(wvout, binary=True)
			self.wordvec = wv
		else:
			self.wordvec = Word2Vec.load_word2vec_format(wf or self.wordvecf, binary=True)

		self.size = self.wordvec.vector_size
		self.originalsize = self.wordvec.vector_size
		self.emptyvec = Word.create("", np.array([1e-9] * self.size))
		self.__logdf = None


class Glove(object):
	def __init__(self, text=None, vocab_file=None, vectors_file=None):

    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]

		allvocab = Counter()
		if text:
			for sentence in text:
				allvocab.update(sentence)
		else:
			allvocab.update(words)

    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
						if words[vals[0]] in allvocab:
            	vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx if w in allvocab for idx, w in enumerate(words)}
    ivocab = {idx: w if w in allvocab for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    self.W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    self.W_norm = (W.T / d).T
		self.vocab = vocab
		self.ivocab = ivocab

	def __getitem__(self, k):
		return self.vocab[k]


def shingle(stoks, ttoks, slop=12, lmbda=0.95):
	k = 3
	scores = []

	for st in range(0, len(stoks) - k):
		pls = [PList(gindex(ttoks, stoks[st + ind]), ind) for ind in range(0, min(k, len(stoks)))]
		validp = True
		for pl in pls:
			if not pl:
				validp = False
				break
		if not validp:
			continue
		ps = PQueue(pls)
		bestspan = ps.span()
		while ps:
			pl = ps.pop()
			if not pl.shift():
				break
			ps.push(pl)
			span = ps.span()
			if span[0] < bestspan[0]:
				bestspan = span

		if bestspan[0] < slop:
			scores.append(bestspan)

	if scores:
		st = min([x[1] for x in scores])
		nd = max([x[2] for x in scores])
		#print scores
		score = sum([lmbda ** ((x[0] - k) / k) for x in scores]) / max(1, len(stoks) - k + 1)
		#print score, len(stoks)
		#sys.exit()
		return Match(score, st, nd)
	else:
		return Match(0, 0, 0)


@functools.total_ordering
class PList(object):
	def __init__(self, l=[], _id=0):
		self.l = l
		self.pos = 0
		self._id = _id

	def __eq__(self, o):
		return self.get() == o.get()

	def __lt__(self, o):
		return self.get() < o.get()

	def __nonzero__(self):
		return len(self.l) > 0

	def get(self):
		return self.l[self.pos]

	def shift(self):
		if self.pos + 1 == len(self.l):
			return False
		self.pos += 1
		return True


class PQueue(object):
	def __init__(self, l=[]):
		heapq.heapify(l)
		self.h = l

	def pop(self):
		return heapq.heappop(self.h)

	def push(self, i):
		return heapq.heappush(self.h, i)

	def __nonzero__(self):
		return len(self.h) > 0

	def span(self):
		vals = [x.get() for x in self.h]
		ret = [0, min(vals), max(vals)]
		ret[0] = ret[2] - ret[1] + 1
		return ret


def gindex(s, ch):
	return [i for i, ltr in enumerate(s) if ltr == ch]


def tokenize(t):
	global stopl
	try:
		toks = word_tokenize(t)
	except Exception:
		toks = t.strip().split()
	toks = [x.lower() for x in toks]
	if not stopl:
		stopl = set(stopwords.words('english'))
	return [x for x in toks if x not in stopl]


def stem(ts):
	global stemmer
	if stemmer is None:
		stemmer = PorterStemmer()
	if type(ts) is list:
		return [stemmer.stem(x) for x in ts]
	else:
		return stemmer.stem(ts)
