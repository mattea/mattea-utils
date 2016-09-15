#!/usr/bin/python
from __future__ import division
import csv
import argparse
import ConfigParser
import codecs
import json
import sys
import os
import time
import heapq
import numpy as np
from functools import total_ordering
#from gensim.models import Word2Vec
#from scipy.spatial import distance
from matteautils.match.vectorsim import MinDistSim, InfSim, VecSim
from matteautils.parser import WordVec
from matteautils.base import printd
from matteautils.dataset import Dataset
import matteautils.config as conf

csv.field_size_limit(sys.maxsize)
sys.stdout = codecs.getwriter('utf8')(sys.stdout)

#wordvecf = '/huge1/sourcedata/word2vec/google/vectors-skipgram1000-en.bin'
wordvecf = '/huge1/sourcedata/word2vec/google/GoogleNews-vectors-negative300.bin'
emptyvec = [1e-10] * 1000
sim_thr = -10
wordvec = []

min_score = 0.1

#if sys.stdout.encoding is None:
#	os.putenv("PYTHONIOENCODING", 'UTF-8')
#	os.execv(sys.executable, ['python'] + sys.argv)


class FeatureWriter(object):

	def writeheader(self):
		pass
		print >>self.sh, "\t".join(("Label", "MatchScore", "WV1Min", "WV1Max", "WV1Sum", "WV1Avg", "WV2Min", "WV2Max", "WV2Sum", "WV2Avg"))

	def write(self, pair, match):
		nugget = pair.s1
		update = pair.s2
		qid = nugget["query_id"]
		print >>self.sh, "\t".join((qid, update["id"], nugget["id"],
																str(match.start), str(match.end),
																str(match.autop), "%g" % match.score,
																update["text"], nugget["text"]))

#
#
#def load_wordvec():
#	return Word2Vec.load_word2vec_format(wordvecf, binary=True)


class Timer(object):
	def __init__(self):
		self.s = time.clock()

	def mark(self):
		s = self.s
		self.s = time.clock()
		return self.s - s


class Match(object):
	def __init__(self, score=0.0, start=0, end=0, autop=1):
		self.score = score
		self.start = start
		self.end = end
		self.autop = autop


@total_ordering
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


#def get_sentvec(words):
#	sent_vec = []
#	witer = iter(range(len(words)))
#	for i in witer:
#		wvec = emptyvec
#		if (i + 1) < len(words):
#			word = words[i] + '_' + words[i + 1]
#			nwvec = get_wordvec(word)
#			if nwvec is not emptyvec:
#				wvec = nwvec
#				witer.next()
#		if wvec is emptyvec:
#			word = words[i]
#			wvec = get_wordvec(word)
#			if wvec is emptyvec:
#				word = stemmer.stem(word)
#				wvec = get_wordvec(word)
#		sent_vec.append(wvec)
#	#return np.sum(sent_vec, axis=0)
#	return sent_vec


#def tonumber(word):
#	try:
#		return float(word)
#	except Exception:
#		return None


#def get_wordvec(word):
#	num = tonumber(word)
#	if num is not None:
#		return WordVec(num)
#	word = '/en/' + word
#	try:
#		return WordVec(wordvec[word])
#	except KeyError:
#		#printd("No vector for " + word)
#		return emptyvec


#def main(nf, uf, sf, vf):
def main(args):
	global wordvec, wordvecf
	conf.debug = args.debug or args.verbose
	conf.verbose = args.verbose
	conf.args = args
	#nf = args.nuggets
	#uf = args.updates
	#mf = args.matches
	sf = args.shingles
	vf = args.wordvec
	#ef = args.evalfile
	wvout = args.wvfile
	sim_thr = args.sim_thr
	dset = args.dataset
	limit = args.limit

	#if args.dataset == "auto":
	#	if ef is not None:
	#		dset = "semeval"
	#	else:
	#		with open(glob.glob(nf)[0]) as nh:
	#			nfhead = nh.readline()
	#			if nfhead.startswith("query_id\tnugget_id"):
	#				dset = "ts"
	#			elif nfhead.startswith("query_id\tvs_id"):
	#				dset = "mclick"
	#			else:
	#				dset = "1click"

	if os.path.exists(wvout) and not args.force:
		wordvecf = wvout

	if vf:
		printd("Reading word vector...")
		#wordvec = load_wordvec()
		wordvec = WordVec(wordvecf)

	if args.sim == "minsim":
		matcher = MinDistSim
	elif args.sim == "infsim":
		matcher = InfSim
	else:
		matcher = VecSim

	if args.sim == "infsim" or args.comparator == "infsim":
		wordvec.normalize()

	#if dset == "ts":
	#	nuggfn = Nuggets
	#	updfn = Updates
	#	outfn = MatchWriter
	#elif dset == "1click":
	#	nuggfn = CLNuggets
	#	updfn = CLUpdates
	#	outfn = CLMatchWriter
	#elif dset == "mclick":
	#	nuggfn = MCNuggets
	#	updfn = Updates
	#	outfn = MCMatchWriter
	#elif dset == "semeval":
	#	data = SemEvalDataset(args.input_data, args.evalfile)
	#	outfn = data.writer
	#	if vf is not None:
	#		data.vectorize(wordvec)
	#else:
	#	nuggfn = MCNuggets
	#	updfn = Updates
	#	outfn = MCMatchWriter

	data = Dataset.load(args.input_data, dset)
	if vf is not None:
		data.vectorize(wordvec)
	#if dset == "semeval":
	#	data = SemEvalDataset(args.input_data, args.evalfile)
	#	#outfn = data.writer
	#	if vf is not None:
	#		data.vectorize(wordvec)
	#else:
	#	printd("Processing Nuggets...")
	#	#nuggets = nuggfn(nf, vectorize=vf is not None)

	#	printd("Processing Updates...")
	#	#updates = updfn(uf, vectorize=vf is not None)
	#	#data = NuggetDataset(nuggets, updates, mf)
	#	data = NuggetDataset(nf, uf, mf, dset=dset, vectorize=vf is not None)

	if vf and wvout is not None and wvout != wordvecf:
		printd("Rereading word vectors to optimize...")
		wv_toks = data.wv_sentences()
		#if dset == "semeval":
		#	wv_toks = data.wv_sentences()
		#else:
		#	wv_toks = nuggets.wv_text() + updates.wv_text()
		wordvec = WordVec(wordvecf, sentences=wv_toks, wvout=wvout, size=wordvec.originalsize)
		if args.sim == "infsim" or args.comparator == "infsim":
			wordvec.normalize()
		data.vectorize(wordvec)
		with open(wvout + ".vocab", 'w') as wh:
			wh.write("\n".join(wordvec.vocab().keys()))
		with open(wvout + ".toks", 'w') as wh:
			wh.write("\n".join([" ".join(x) for x in wv_toks]))
		#vocab = nuggets.wv_vocab().union(updates.wv_vocab())
		#wordvec.trim(lambda word, count, min_count: gensim.utils.RULE_KEEP if word in vocab else gensim.utils.RULE_DISCARD)
		#wordvec.save(wvout)

	vocab = None
	if args.frequencies:
		try:
			with open(args.frequencies) as fh:
				vocab = json.load(fh)
			# For Term Frequencies instead of Document Frequencies
			# Could also do len(vocab[word]) if wanted to mimic DF
			if type(vocab.itervalues().next()) == dict:
				for word in vocab:
					vocab[word] = sum(vocab[word].itervalues())
		except Exception:
			pass
	if vocab is None:
		vocab = data.wv_vocab()
	logdf = wordvec.logdf(vocab)
	logdffile = wordvecf + ".logdf"
	#if not os.path.exists(logdffile) or (os.path.getmtime(logdffile) < os.path.getmtime(wordvecf)):
	#	np.savetxt(logdffile, logdf, delimiter=" ", fmt="%g")
	np.savetxt(logdffile, logdf, delimiter=" ", fmt="%g")

	if args.comparator == "infsim" and args.sim != "infsim":
		comparator = InfSim(logdf).pairwisedist
	else:
		comparator = args.comparator

	matcher = matcher(df=logdf, metric=comparator)
	data.normalize(matcher, logdf)

	printd("Finding matches...")
	matches = []
	with data.writer(sf) as sw, data.writer(vf) as vw:
		mcnt = 0
		timer = Timer()
		for pair in data.test():
			if sf:
				match = shingle(pair.s1["tokens"], pair.s2["tokens"])
				if match.score >= min_score:
					sw.write(pair, match)

			if vf:
				printd("Matching pair %s" % (pair.pid), level=1)
				try:
					sim = matcher.match(pair)
					matches.append((matcher.tsim, unicode(matcher)))
				except ValueError, err:
					printd(err)
					sim = sim_thr
				printd("Match %0.4f for %s, %s" % (sim, pair.sid1, pair.sid2))
				if sim < sim_thr:
					sim = sim_thr
					start = matcher.start
					end = matcher.end - matcher.start
				else:
					start = -1
					end = len(pair.s2["tokens"]) - 1
				match = Match(sim, start, end)
				vw.write(pair, match)

			mcnt += 1
			if (mcnt % 100000) == 0:
				print >>sys.stderr, "%g tmps" % (100 / timer.mark())
			if limit and mcnt >= limit:
				return

		if conf.verbose:
			for tsim, match in sorted(matches):
				print match


#if __name__ == '__main__':
def cmdline():
	conf_parser = argparse.ArgumentParser(
		# Turn off help, so we print all options in response to -h
		add_help=False
		)
	conf_parser.add_argument("-c", "--conf_file",
		help="Specify config file", metavar="FILE")
	args, remaining_argv = conf_parser.parse_known_args()
	defaults = {
		"input_data": "../results/nuggets.tsv",
		"dataset": "auto",
		"sim": "sum",
		"comparator": "cosine",
		"sim_thr": -10,
		"limit": None,
		}
	if args.conf_file:
		config = ConfigParser.SafeConfigParser()
		config.read([args.conf_file])
		defaults = dict(config.items("Defaults"))

	argparser = argparse.ArgumentParser(description='Automatically match using shingles and word vectors', parents=[conf_parser])
	argparser.set_defaults(**defaults)
	argparser.add_argument('-i', '--input_data', help='Dataset file/dir')
	argparser.add_argument('-n', '--nuggets', help='Nuggets file/dir')
	argparser.add_argument('-u', '--updates', help='Updates file/dir')
	argparser.add_argument('-m', '--matches', help='Matches file/dir')
	argparser.add_argument('-d', '--debug', action='store_true', help='Debug mode (lots of output)')
	argparser.add_argument('--verbose', action='store_true', help='Even more output')
	argparser.add_argument('-e', '--evalfile', help='Single Eval file (e.g. SICK)')
	argparser.add_argument('-s', '--shingles', help='Write out shingles to file/dir')
	argparser.add_argument('-v', '--wordvec', help='Write out wordvec to file/dir')
	argparser.add_argument('--frequencies', help='Read from word frequencies JSON file')
	argparser.add_argument('-f', '--force', action='store_true', help='Force rewriting word vector output file')
	argparser.add_argument('-w', '--wvfile', help='Store minimal version of wordvec file after processing, ' +
			'for faster subsequent runs (reused if exists, unless -f used).')
	argparser.add_argument('--dataset', help='Type of Input/Output: ts,1click,mclick,semeval,auto (default: try to detect from filenames)')
	argparser.add_argument('--sim', help='Similarity calculation to use, "minsim", "infsim" or "sum"')
	argparser.add_argument('--learn', help='Write features for a learning algorithm')
	argparser.add_argument('--comparator', help='Comparator to use, when appropriate: ' +
		'"cosine" (default), "infsim", or any metric from scipy.spatial.distance.cdist')
	argparser.add_argument('--sim_thr', type=float, help='Similarity threshold to use')
	argparser.add_argument('--limit', type=int, help='Limit number of matches performed (for testing)')
	main(argparser.parse_args(remaining_argv))
