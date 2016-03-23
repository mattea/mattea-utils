from peak.util.imports import lazyModule
import functools
stemmer = None
word_tokenize = lazyModule('nltk.word_tokenize')
PorterStemmer = lazyModule('nltk.stem.porter.PorterStemmer')
from nltk.corpus import stopwords
stopl = None
from gensim.models import Word2Vec


class Match(object):
	def __init__(self, score=0.0, start=0, end=0, mtype=None):
		self.score = score
		self.start = start
		self.end = end
		self.mtype = mtype


wvsource="https://docs.google.com/uc?export=download&confirm=xpiI&id=0B7XkCwpI5KDYeFdmcVltWkhtbmM"


class WordVec(object):
	wordvecf = '/huge1/sourcedata/word2vec/google/vectors-skipgram1000-en.bin'
	emptyvec = [1e-10] * 1000
	wordvec = []

	def __init__(self, wf=None):
		self.wordvec = Word2Vec.load_word2vec_format(wf or wordvecf, binary=True)

	def get_sentvec(self, words):
		sent_vec = []
		witer = iter(range(len(words)))
		for i in witer:
			wvec = self.emptyvec
			if (i + 1) < len(words):
				word = words[i] + '_' + words[i + 1]
				nwvec = self.get_wordvec(word)
				if nwvec is not self.emptyvec:
					wvec = nwvec
					witer.next()
			if wvec is self.emptyvec:
				word = words[i]
				wvec = self.get_wordvec(word)
			sent_vec.append(wvec)
		return sent_vec
	
	
	def get_wordvec(self, word):
		word = '/en/' + word
		try:
			return self.wordvec[word]
		except KeyError:
			return self.emptyvec


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
	return [stemmer.stem(x) for x in ts]

