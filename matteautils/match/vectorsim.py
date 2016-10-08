#!/usr/bin/python
from __future__ import division
import math
import signal
import sys
import numpy as np
from scipy.spatial import distance
from munkres import munkres
from . import Matcher
from itertools import izip
from scipy.stats import kendalltau
from matteautils.base import printd
import matteautils.config as conf
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from skidmarks import wald_wolfowitz


def munkres_handler(signum, frame):
	printd("Can't keep waiting...")
	print frame
	raise Exception("ran out of time...")

#a = np.array([[.1]*5,[.2]*5,[.3]*5])
#b = np.array([[.1]*5,[.2,.2,.2,.2,.3],[0,0,0,0,1]])
#c = 1 - cdist(a,b,'cosine')


def multiCdist(s1, s2, metric):
	if s1.ndims == 1:
		return metric(s1["vector"], s2["vector"])
	else:
		return sum([metric(x, y) * a for x, y, a in izip(s1["vector"], s2["vector"], conf.alphas)])


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
		self.pair = pair
		self.s1 = pair.s1
		self.s2 = pair.s2
		self.tsim = 1 - self.metric([self.s1["vec_sum"]], [self.s2["vec_sum"]])[0, 0]
		self.nmatches = -1
		self.start = -1
		self.end = len(self.s2["vector"]) - 1
		return self.tsim


class MinDistSim(Matcher):

	def __init__(self, df=None, metric='cosine', maxsent=20, ngram=1, recurse=False, dimfeatures=True):
		#self.dist = ndist(s1, s2)
		#s1 = s1["vector"]
		#s2 = s2["vector"]
		self.metric = getMetric(metric)
		self._names = ["MDS_" + x for x in ["tsim", "lsim", "kdist", "kldist", "ldist", "kt", "tmax", "tmin", "tsum", "tstd", "tmaxidf", "tsumidf"]]
		maxsent = maxsent - ngram + 1
		if dimfeatures:
			self._names.extend(["MDS_w%03d" % x for x in range(maxsent)])
		self.maxsent = maxsent
		self.ngram = ngram
		self.recurse = recurse
		self.vocab = df
		self.wordcount = df.total
		self.dimfeatures = dimfeatures

	def match(self, pair):
		s1l = len(pair.s1["vector"])
		s2l = len(pair.s2["vector"])
		self.tsim = float('-9999')
		self.lsim = float('-9999')
		self.minlen = min(s1l, s2l)
		self.maxlen = max(s1l, s2l)
		self.nmatches = 0
		self.start = -1
		self.end = -1
		if (self.minlen == 0 or
				self.maxlen >= 100):
			return self.tsim

		# For simplicity in later code, make the shorter one first
		if s1l < s2l:
			self.s1 = pair.s1
			self.s2 = pair.s2
			s1l = len(pair.s1["vector"])
			s2l = len(pair.s2["vector"])
		else:
			self.s1 = pair.s2
			self.s2 = pair.s1

		wc = self.wordcount
		if "wv_idfs" not in self.s1:
			self.s1["wv_idfs"] = [math.log(wc / self.vocab[x], 2) for x in self.s1["wv_tokens"]]
		if "wv_idfs" not in self.s2:
			self.s2["wv_idfs"] = [math.log(wc / self.vocab[x], 2) for x in self.s2["wv_tokens"]]

		if self.ngram > 1:
			ng = self.ngram
			v1 = self.s1["vector"]
			v2 = self.s2["vector"]
			t1 = self.s1["wv_tokens"]
			t2 = self.s2["wv_tokens"]
			#idf1 = self.s1["wv_idfs"]
			#idf2 = self.s2["wv_idfs"]
			weights1 = self.s1["weights"]
			weights2 = self.s2["weights"]
			nv1 = [sum(v1[i:i + ng]) for i in range(max(1, len(v1) - ng + 1))]
			nv2 = [sum(v2[i:i + ng]) for i in range(max(1, len(v2) - ng + 1))]
			nt1 = ["_".join(t1[i:i + ng]) for i in range(max(1, len(t1) - ng + 1))]
			nt2 = ["_".join(t2[i:i + ng]) for i in range(max(1, len(t2) - ng + 1))]
			#nidf1 = [max(idf1[i:i + ng]) for i in range(max(1, len(idf1) - ng + 1))]
			#nidf2 = [max(idf2[i:i + ng]) for i in range(max(1, len(idf2) - ng + 1))]
			nweights1 = [max(weights1[i:i + ng]) for i in range(max(1, len(weights1) - ng + 1))]
			nweights2 = [max(weights2[i:i + ng]) for i in range(max(1, len(weights2) - ng + 1))]
			#self.s1 = {"vector": nv1, "wv_tokens": nt1, "wv_idfs": nidf1}
			#self.s2 = {"vector": nv2, "wv_tokens": nt2, "wv_idfs": nidf2}
			self.s1 = {"vector": nv1, "wv_tokens": nt1, "weights": nweights1}
			self.s2 = {"vector": nv2, "wv_tokens": nt2, "weights": nweights2}

			self.minlen = max(self.minlen - ng + 1, 1)
			self.maxlen = max(self.maxlen - ng + 1, 1)

		self.dists = [1] * self.minlen

		self.pair = pair
		#self.dist = pairdist(self.s1["vector"], self.s2["vector"], fn=self.metric)
		#self.dist = pairdist(self.s1, self.s2, fn=self.metric)
		dist = self.metric(self.s1, self.s2)

		# scale by max of idf
		#for i in range(dist.shape[0]):
		#	for j in range(dist.shape[1]):
		#		dist[i][j] *= max(self.s1["wv_idfs"][i], self.s2["wv_idfs"][j])

		self.matchv = np.zeros(dist.shape, int)
		np.fill_diagonal(self.matchv, 1)
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
		signal.alarm(10)
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
		tmaxidf = 0
		tsumidf = 0
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
		dists = [0] * matches.shape[0]
		matchedy = [0] * matches.shape[1]
		for i in range(matches.shape[0]):
			for j in range(matches.shape[1]):
				if matches[i, j]:
					matchedy[j] = 1
					tdist += dist[i, j]
					#tmaxidf += dist[i, j] * max(self.s1["wv_idfs"][i], self.s2["wv_idfs"][j])
					#tsumidf += dist[i, j] * sum((self.s1["wv_idfs"][i], self.s2["wv_idfs"][j]))
					wi = self.s1["weights"][i]
					wj = self.s2["weights"][j]
					tmaxidf += dist[i, j] * max(wi, wj)
					tsumidf += dist[i, j] * sum((wi, wj))
					printd("%s\t%s\t%0.4f\t%0.4f\t%0.4f" % (s1tok[i], s2tok[j], dist[i, j], wi, wj), level=1, sock=sys.stdout)
					nmatches += 1
					matcharr[i] = j
					dists[i] = dist[i, j]
					if j < mstart:
						mstart = j
					if j > mend
		tsumidf = tsumidf * max(dist.shape) / pow(min(dist.shape), 2)
		kt, ktp = kendalltau(range(len(matcharr)), matcharr)
		printd("Score: %0.4f\t%0.4f\t%0.4f\tLabel: %g\n" % (tdist, tmaxidf, tsumidf, pair.label), level=1, sock=sys.stdout)
		if self.recurse:
			# Remove matches from dist array, and rerun munkres
			# Repeat until dist array is empty
			pass
		else:
			for i in range(matches.shape[1]):
				if not matchedy[i]:
					ldist += min(matches[:, i])
		ldist /= max(dist.shape)
		# TODO:
		# Dist penalty is at most beta
		# The problem with this is that there may be a better pairing between the two sentences
		# if you optimize for mindist with dist penalty.
		# Also could add a weight to each pairing like IDF, most important for the
		# summing, but a different sum would again potentially affect the optimal
		# match.
		beta = 1
		self.kdist = tdist * (1 + beta * (kt + 1) / 2)
		self.kldist = ldist * (1 + beta * (kt + 1) / 2)
		self.ldist = ldist
		#print "Score: %g" % tsim
		#print "Label: %g" % self.pair.label
		self.tsim = 1 - tdist
		self.tmaxidf = tmaxidf
		self.tsumidf = tsumidf
		self.nmatches = nmatches
		self.start = mstart
		self.end = mend
		self.kt = kt
		self.dists = sorted(dists, reverse=True)
		self.lsim = tdist + (max(dists) * (self.maxlen - self.minlen))
		self.tmax = max(dists)
		self.tmin = max(dists)
		self.tsum = sum(dists)
		self.tstd = np.std(dists)
		return self.tsim

	def features(self):
		fs = [self.tsim, self.lsim, self.kdist, self.kldist, self.ldist, self.kt, self.tmax, self.tmin, self.tsum, self.tstd, self.tmaxidf, self.tsumidf]
		if self.dimfeatures:
			distarr = [0] * self.maxsent
			dists = self.dists
			distarr[0:len(dists)] = dists
			fs += distarr
		return fs


class InfSim(Matcher):
	#def __init__(self, df, metric='cosine'):
	#	#self.metric = getMetric(metric)
	#	self.df = df

	#	self._vectorsums = dict()

	def __init__(self, data, wordvec, metric='cosine', dimfeatures=False):
		self.df = wordvec.logdf(data.wv_vocab())
		data.normalize(self, self.df)

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
		dists = np.zeros((len(s1["wv_tokens"]), len(s2["wv_tokens"])))
		for wi1, (w1, v1) in enumerate(izip(s1["wv_tokens"], s1["vector"])):
			for wi2, (w2, v2) in enumerate(izip(s2["wv_tokens"], s2["vector"])):
				self.s1 = v1
				self.s2 = v2
				self.ts1 = self.vectorsum(w1, v1)
				self.ts2 = self.vectorsum(w2, v2)
				dists[wi1, wi2] = 1 - self.domatch()
				#TODO could multiply by term based on wi1/wi2 (distance penalty)...
				if dists[wi1, wi2] < 0:
					dists[wi1, wi2] = 0
					#printd("Hmm, negative dist %g" % dists[wi1, wi2])
					# Annoying rounding errors, e.g. -2.22045e-16
		return dists

	def vectorsum(self, word, wv):
		if word not in self._vectorsums:
			self._vectorsums[word] = np.sum(np.multiply(wv, self.df))
		return self._vectorsums[word]


class InfRankSim(Matcher):

	def __init__(self, data, wordvec, df=None, metric='cosine', dimfeatures=True):
		self._vectorsums = dict()
		self.vocabvec = self.sortwords(data.wv_vocab(), wordvec)
		self._names = ["IRS_" + x for x in ["asim", "mdim", "msent"]]
		if dimfeatures:
			self._names.extend(["IRS_d%03d" % x for x in range(wordvec.size)])
		self.dimfeatures = dimfeatures

	def sortwords(self, vocab, wordvec):
		vvecs = [list() for _ in range(wordvec.size)]
		ftot = 0
		for t, tc in vocab.iteritems():
			try:
				tv = wordvec[t]
				ftot += tc
				for d in range(len(tv)):
					vvecs[d].append((tv[d], t))
			except KeyError:
				pass

		lookupvecs = [dict() for _ in range(wordvec.size)]
		for d, vvec in enumerate(vvecs):
			vvec.sort()
			cdf = 0
			#vtot = len(vvec)
			#vitm = 1 / vtot
			lookup = lookupvecs[d]
			for tv, t in vvec:
				# Should the CDF be based on TF? or just on the word existence?
				cdf += tv / ftot
				#cdf += vitm
				lookup[t] = cdf

		return lookupvecs

	def match(self, pair):
		wvlen = len(pair.s1["vector"][0])
		m = len(pair.s1["wv_tokens"])
		n = len(pair.s2["wv_tokens"])
		self._features = []

		# Take the average of all distances
		asim = 0
		mdim = 0
		msent = 0
		for d in range(wvlen):
			mindimp = 1
			for t1 in pair.s1["wv_tokens"]:
				minsentp = 1
				for t2 in pair.s2["wv_tokens"]:
					p = abs(self.vocabvec[d][t1] - self.vocabvec[d][t2])
					asim += simP(p)

					if p < mindimp:
						mindimp = p
					if p < minsentp:
						minsentp = p

				# Take the minimum across one sentences
				msent += simP(minsentp)

			# Take the minimum across dimensions
			mdim += simP(mindimp)

		asim /= m * n * wvlen
		self._features.append(asim)

		self._features.append(mdim / wvlen)

		msent /= n * wvlen
		self._features.append(msent)

		if self.dimfeatures:
			for d in range(wvlen):
				combvec = ([(self.vocabvec[d][t1], 0) for t1 in pair.s1["wv_tokens"]] +
										[(self.vocabvec[d][t2], 1) for t2 in pair.s1["wv_tokens"]])
				combvec.sort()
				combval = multicdf(combvec)
				self._features.append(simP(combval))

		self.tsim = asim
		self.start = 0
		self.end = m


def simP(p):
	if p == 0:
		return 1
	elif p == 1:
		return 0
	else:
		return 1 / (1 + (1 / (math.log(1 / p, 2))))


# TODO: Should add slack, so not all values must match. Would require a lot
# more bookkeeping
def multicdf(vec):
	lastvs = [[], []]
	cdf = 0
	for v, senti in vec:
		vs = lastvs[senti]
		ovs = lastvs[senti - 1]
		if len(ovs) != 0:
			if len(vs) == 0:
				cdf += v - ovs[0]
				del ovs[:]
			else:
				back = None
				forward = None
				prevv = vs[0]
				# If expecting large set, could do binary search...
				for ov in ovs:
					if (ov - prevv) < (v - ov):
						back = ov
					else:
						forward = ov
						break
				if back is not None:
					cdf += back - prevv
				if forward is not None:
					cdf += v - forward
				del vs[:]
				del ovs[:]

		vs.append(v)

		#if lasti is not None:
		#	cdf += (v - lastv)
		#	if senti != lasti:
		#		lasti = None
		#else:
		#	lasti = senti
		#lastv = v

	return cdf


def ndist(s1, s2, fn='cosine'):
	rd = []
	for w1 in s1:
		rr = []
		for w2 in s2:
			rr.append(w1.dist(w2, fn=fn))
		rd.append(rr)
	return np.array(rd)


# Wald-Wolfowitz test
# Adapted from:
# Monaco, John V.
# "Classification and authentication of one-dimensional behavioral biometrics."
# Biometrics (IJCB), 2014 IEEE International Joint Conference on. IEEE, 2014.
# https://gist.github.com/vmonaco/e9ff0ac61fcb3b1b60ba
class WWSim(Matcher):
	def __init__(self, wordvec, df=None, metric='cosine', k=10, dimfeatures=True):
		self.k = 10
		self._names = ["WWS_base"]
		if dimfeatures:
			self._names.extend(["WWS_%03d" % x for x in range(wordvec.size)])
		self.dimfeatures = dimfeatures

	def match(self, pair):
		self.pair = pair
		v1 = pair.s1["vector"]
		v2 = pair.s2["vector"]
		m = len(v1)
		n = len(v2)
		N = m + n
		k = min(N - 1, self.k)
		if m == 0 or n == 0 or np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
			return 0

		vs = np.concatenate((v1, v2))
		g = kneighbors_graph(vs, mode='distance', n_neighbors=k)
		mst = minimum_spanning_tree(g, overwrite=True)
		edges = np.array(mst.nonzero()).T
		labels = np.array([0] * m + [1] * n)

		c = labels[edges]
		runs_edges = edges[c[:, 0] == c[:, 1]]

		# number of runs is the total number of observations minus edges within each run
		R = N - len(runs_edges)

		# expected value of R
		e_R = ((2.0 * m * n) / N) + 1

		# variance of R is _numer/_denom
		v = 2 * m * n * (2 * m * n - N) / (N ** 2 * (N - 1))

		# see Eq. 1 in Friedman 1979
		# W approaches a standard normal distribution
		W = (R - e_R) / np.sqrt(v)

		self.tsim = -1 if np.isnan(W) else W

		bydim = []
		for d in range(len(v1[0])):
			sorteddim = np.argsort(vs[:, d])
			wd = wald_wolfowitz(labels[sorteddim])
			bydim.append(wd['z'])

		self._features = [self.tsim]
		if self.dimfeatures:
			self._features += bydim

		return self.tsim


class PairFeatures(Matcher):
	def __init__(self, dimfeatures=True):
		self.dimfeatures = dimfeatures
		self._names = ["PF_" + x for x in ["Len", "TextLen", "TextDiff"]]

	def match(self, pair):
		fs = []
		self._features = fs
		self.tsim = abs(len(pair.s1["vector"]) - len(pair.s2["vector"]))
		fs.append(self.tsim)
		fs.append(abs(len(pair.s1["text"]) - len(pair.s2["text"])))
		fs.append(abs(len(pair.s1["text"]) + len(pair.s2["text"]) - len(" ".join(pair.s1["wv_tokens"])) - len(" ".join(pair.s2["wv_tokens"]))))
		return self.tsim
