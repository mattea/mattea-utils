#!/usr/bin/python
import numpy as np
from scipy.spatial import distance
from munkres import munkres
from . import Matcher
from itertools import izip
import sys
import signal


def munkres_handler(signum, frame):
	printd("Can't keep waiting...")
	print frame
	raise Exception("ran out of time...")

#a = np.array([[.1]*5,[.2]*5,[.3]*5])
#b = np.array([[.1]*5,[.2,.2,.2,.2,.3],[0,0,0,0,1]])
#c = 1 - cdist(a,b,'cosine')


def printd(s):
	print >>sys.stderr, s


class VecSim(Matcher):

	def __init__(self, pair, df=None):
		self.s1 = pair.s1
		self.s2 = pair.s2

	def match(self):
		self.tsim = 1 - distance.cosine(self.s1["vec_sum"], self.s2["vec_sum"])
		self.nmatches = -1
		self.start = -1
		self.end = len(self.s2["sent_vec"]) - 1
		return self.tsim


class MinSim(Matcher):

	def __init__(self, pair, df=None):
		#self.dist = ndist(s1, s2)
		#s1 = s1["sent_vec"]
		#s2 = s2["sent_vec"]
		self.dist = dist(pair.s1["sent_vec"], pair.s2["sent_vec"])
		self.s1 = pair.s1
		self.s2 = pair.s2
		self.pair = pair
		self.minlen = min(self.dist.shape)
		self.maxlen = max(self.dist.shape)

	def match(self):
		dist = self.dist
		self.matchv = np.zeros(dist.shape, int)
		np.fill_diagonal(self.matchv, 1)
		if (min(dist.shape) == 0 or
				max(dist.shape) >= 100):
			self.tsim = float('-inf')
			self.start = -1
			self.end = -1
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
			self.tsim = float('-inf')
			self.start = -1
			self.end = -1
			return self.tsim
		signal.alarm(0)
		self.matchv = matches

		tsim = 0
		nmatches = 0
		mstart = dist.shape[1]
		mend = 0
		print self.s1["tokens"]
		print self.s2["tokens"]
		print self.s1["wv_tokens"]
		print self.s2["wv_tokens"]
		s1tok = self.s1["wv_tokens"]
		s2tok = self.s2["wv_tokens"]
		for i in range(matches.shape[0]):
			for j in range(matches.shape[1]):
				if matches[i, j]:
					tsim += self.dist[i, j]
					print "%s\t%s\t%0.4f" % (s1tok[i], s2tok[j], self.dist[i, j])
					nmatches += 1
					if j < mstart:
						mstart = j
					if j > mend:
						mend = j
		tsim = tsim * max(dist.shape) / pow(min(dist.shape), 2)
		print "Score: %g" % tsim
		print "Label: %g" % self.pair.label
		self.tsim = 1 - tsim
		self.nmatches = nmatches
		self.start = mstart
		self.end = mend
		return tsim


class InfSim(Matcher):
	def __init__(self, pair, df):
		self.s1 = pair.s1["sent_vec"]
		self.s2 = pair.s2["sent_vec"]
		self.ts1 = pair.s1["sent_sum"]
		self.ts2 = pair.s2["sent_sum"]
		self.df = df

	@classmethod
	def normalize(cls, s, df):
		if len(s) == 0:
			return s, 0
		if np.any(np.isnan(df)):
			printd("Hmm, nan for df %0.4f")
			printd("df:\n" + str(df))
			sys.exit(1)
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

	def match(self):
		if self.ts1 == 0 or self.ts2 == 0:
			self.tsim = float('-inf')
			self.start = -1
			self.end = -1
			return self.tsim
		self.tsim = 2 * sum([min(s1i, s2i) * dfi for s1i, s2i, dfi in izip(self.s1, self.s2, self.df)]) / (self.ts1 + self.ts2)
		self.nmatches = -1
		self.start = -1
		self.end = len(self.s2) - 1
		return self.tsim


# Lots of possible fns:
# http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.spatial.distance.cdist.html
def dist(s1, s2, fn='cosine'):
	if len(s1) == 0 or len(s2) == 0:
		return np.ndarray(0)
	else:
		if type(s1[0]) == np.ndarray:
			return distance.cdist(s1, s2, fn)
		else:
			return distance.cdist([x.vector for x in s1], [x.vector for x in s2], fn)


def ndist(s1, s2, fn='cosine'):
	rd = []
	for w1 in s1:
		rr = []
		for w2 in s2:
			rr.append(w1.dist(w2, fn=fn))
		rd.append(rr)
	return np.array(rd)
