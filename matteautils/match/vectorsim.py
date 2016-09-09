#!/usr/bin/python
import numpy as np
from scipy.spatial import distance
from munkres import munkres
from . import Matcher
from itertools import izip
import sys
import signal
from scipy.stats import kendalltau
from matteautils.base import printd


def munkres_handler(signum, frame):
	printd("Can't keep waiting...")
	print frame
	raise Exception("ran out of time...")

#a = np.array([[.1]*5,[.2]*5,[.3]*5])
#b = np.array([[.1]*5,[.2,.2,.2,.2,.3],[0,0,0,0,1]])
#c = 1 - cdist(a,b,'cosine')


def getMetric(metric):
	if isinstance(metric, basestring):
		if metric in locals():
			return locals()[metric]
		else:
			return lambda x, y: distance.cdist(x["vector"], y["vector"], metric=metric)
	else:
		return metric


# Lots of possible fns:
# http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.spatial.distance.cdist.html
def pairdist(s1, s2, distfn=getMetric('cosine')):

	if type(s1[0]) == np.ndarray:
		if len(s1) == 0 or len(s2) == 0:
			return np.ndarray(0)
		return distfn(s1, s2)
	else:
		return distfn(s1["vector"], s2["vector"])
		#return distfn([x.vector for x in s1], [x.vector for x in s2])


class VecSim(Matcher):

	def __init__(self, df=None, metric='cosine'):
		self.metric = getMetric(metric)

	@classmethod
	def normalize(self, vec, df=None):
		return vec, np.sum(vec, axis=0)

	def match(self, pair):
		self.s1 = pair.s1
		self.s2 = pair.s2
		self.tsim = 1 - self.metric([self.s1["vec_sum"]], [self.s2["vec_sum"]])[0, 0]
		self.nmatches = -1
		self.start = -1
		self.end = len(self.s2["vector"]) - 1
		return self.tsim


class MinDistSim(Matcher):

	def __init__(self, df=None, metric='cosine'):
		#self.dist = ndist(s1, s2)
		#s1 = s1["vector"]
		#s2 = s2["vector"]
		self.metric = getMetric(metric)

	def match(self, pair):
		# For simplicity in later code, make the shorter one first
		if len(pair.s1["vector"]) < len(pair.s2["vector"]):
			self.s1 = pair.s1
			self.s2 = pair.s2
		else:
			self.s1 = pair.s2
			self.s2 = pair.s1
		self.pair = pair
		self.minlen = min(self.dist.shape)
		self.maxlen = max(self.dist.shape)
		#self.dist = pairdist(self.s1["vector"], self.s2["vector"], fn=self.metric)
		#self.dist = pairdist(self.s1, self.s2, fn=self.metric)
		self.dist = self.metric(self.s1, self.s2)

		dist = self.dist
		self.matchv = np.zeros(dist.shape, int)
		np.fill_diagonal(self.matchv, 1)
		self.tsim = float('-inf')
		self.nmatches = 0
		self.start = -1
		self.end = -1
		if (min(dist.shape) == 0 or
				max(dist.shape) >= 100):
			return self.tsim
		if np.sum(dist) == 0:
			self.tsim = 1
			self.nmatches = min(dist.shape)
			self.start = 0
			self.end = dist.shape[1] - 1
			return self.tsim
		if (dist == dist[0]).all():
			self.tsim = 1 - sum(dist[0])
			self.nmatches = min(dist.shape)
			self.start = 0
			self.end = dist.shape[1] - 1
			return self.tsim
		if (dist.T == dist[:, 0]).all():
			self.tsim = 1 - sum(dist[:, 0])
			self.nmatches = min(dist.shape)
			self.start = 0
			self.end = dist.shape[1] - 1
			return self.tsim

		signal.signal(signal.SIGALRM, munkres_handler)
		signal.alarm(60)
		try:
			matches = munkres(dist)
		except Exception, e:
			printd(e)
			printd("dist: " + dist.shape)
			printd(dist)
			return self.tsim
		signal.alarm(0)
		self.matchv = matches

		tdist = 0
		nmatches = 0
		mstart = dist.shape[1]
		mend = 0
		#print self.s1["text"]
		#print self.s2["text"]
		#print " ".join(self.s1["wv_tokens"])
		#print " ".join(self.s2["wv_tokens"])
		s1tok = self.s1["wv_tokens"]
		s2tok = self.s2["wv_tokens"]
		matcharr = [0] * matches.shape[0]
		for i in range(matches.shape[0]):
			for j in range(matches.shape[1]):
				if matches[i, j]:
					tdist += self.dist[i, j]
					print "%s\t%s\t%0.4f" % (s1tok[i], s2tok[j], self.dist[i, j])
					nmatches += 1
					matcharr[i] = j
					if j < mstart:
						mstart = j
					if j > mend:
						mend = j
		tdist = tdist * max(dist.shape) / pow(min(dist.shape), 2)
		kt, ktp = kendalltau(range(len(matcharr)), matcharr)
		# TODO:
		# Dist penalty is at most beta
		# The problem with this is that there may be a better pairing between the two sentences
		# if you optimize for mindist with dist penalty.
		# Also could add a weight to each pairing like IDF, most important for the
		# summing, but a different sum would again potentially affect the optimal
		# match.
		beta = 1
		tdist = tdist * (1 + beta * (kt + 1) / 2)
		#print "Score: %g" % tsim
		#print "Label: %g" % self.pair.label
		self.tsim = 1 - tdist
		self.nmatches = nmatches
		self.start = mstart
		self.end = mend
		return self.tsim

	def __unicode__(self):
		return "\n".join((self.pair.s1["text"],
										self.pair.s2["text"],
										" ".join(self.pair.s1["wv_tokens"]),
										" ".join(self.pair.s2["wv_tokens"]),
										"Score: %g" % self.tsim,
										"Label: %g" % self.pair.label))


class InfSim(Matcher):
	def __init__(self, df, metric='cosine'):
		#self.metric = getMetric(metric)
		self.df = df

		self._vectorsums = dict()

	@classmethod
	def normalize(cls, s, df):
		if len(s) == 0:
			return s, 0
		if np.any(np.isnan(df)):
			printd("Hmm, nan for df %0.4f")
			printd("df:\n" + str(df))
			sys.exit(1)
		# TODO: This should be a weighted sum with IDF
		# As a result of this sum, different length sentences naturally receive a
		# penalty, as the sum is naturally larger than the min.
		# Also, we aren't looking at euclidean distance, so we may be losing out on scale information
		# But if we did, longer sentences would be harder to match together (as distances would compound).
		# Maybe should divide by min sentence legnth or something of the sort...
		#This is avg, not sum...................................................
		# probably causes all sorts of weirdness
		ps = np.sum(s, axis=0) / np.sum(s)
		if np.any(np.isnan(ps)):
			printd("Hmm, nan for ps %0.4f" % np.sum(s))
			printd("ps:\n" + str(ps))
			printd("s:\n" + str(s))
			printd("df:\n" + str(df))
			sys.exit(1)
		ts = np.sum(np.multiply(ps, df))
		if ts == 0:
			printd("Hmm, 0 for ts")
			printd("ps:\n" + str(ps))
			printd("df:\n" + str(df))
			sys.exit(1)
		return ps, ts

	def match(self, pair):
		self.pair = pair
		self.s1 = pair.s1["vector"]
		self.s2 = pair.s2["vector"]
		self.ts1 = pair.s1["vector_sum"]
		self.ts2 = pair.s2["vector_sum"]

		return self.domatch()

	def domatch(self):
		self.nmatches = -1
		if self.ts1 == 0 or self.ts2 == 0:
			self.tsim = 0.0
			self.start = -1
			self.end = -1
			return self.tsim
		self.tsim = 2 * sum([min(s1i, s2i) * dfi for s1i, s2i, dfi in izip(self.s1, self.s2, self.df)]) / (self.ts1 + self.ts2)
		self.start = -1
		self.end = len(self.s2) - 1
		return self.tsim

	def pairwisedist(self, s1, s2):
		#Must create the "vector" and "vector_sum" for each word, rather than for each sentence
		dists = np.zeros((len(s1), len(s2)))
		for wi1, w1, v1 in enumerate(izip(s1["wv_tokens"], s1["vector"])):
			for wi2, w2, v2 in enumerate(izip(s2["wv_tokens"], s2["vector"])):
				self.s1 = v1
				self.s2 = v2
				# Probably should memoize
				self.ts1 = self.vectorsum(w1, v1)
				self.ts2 = self.vectorsum(w2, v2)
				dists[wi1, wi2] = 1 - self.domatch()
		return dists

	def vectorsum(self, word, wv):
		if word not in self._vectorsums:
			self._vectorsums[word] = np.sum(np.multiply(wv, self.df))
		return self._vectorsums[word]

	def __unicode__(self):
		return "\n".join((self.pair.s1["text"],
										self.pair.s2["text"],
										" ".join(self.pair.s1["wv_tokens"]),
										" ".join(self.pair.s2["wv_tokens"]),
										"Score: %g" % self.tsim,
										"Label: %g" % self.pair.label))


def ndist(s1, s2, fn='cosine'):
	rd = []
	for w1 in s1:
		rr = []
		for w2 in s2:
			rr.append(w1.dist(w2, fn=fn))
		rd.append(rr)
	return np.array(rd)
