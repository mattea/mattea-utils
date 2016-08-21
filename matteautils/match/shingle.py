#!/usr/bin/python
from __future__ import division
import csv
import glob
import argparse
import ConfigParser
import codecs
import json
import sys
import os
import time
from collections import defaultdict, Counter
from itertools import chain
import heapq
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from functools import total_ordering
from gensim.models import Word2Vec
from scipy.spatial import distance
from matteautils.match.vectorsim import MinSim, InfSim, VecSim
from matteautils.parser import WordVec
from matteautils.dataset import Dataset
from matteautils.dataset.semeval import SemEvalDataset, SentencePair

csv.field_size_limit(sys.maxsize)

#wordvecf = '/huge1/sourcedata/word2vec/google/vectors-skipgram1000-en.bin'
wordvecf = '/huge1/sourcedata/word2vec/google/GoogleNews-vectors-negative300.bin'
emptyvec = [1e-10] * 1000
sim_thr = -10
#wordvec = []
wordvec = None

debug = True

min_score = 0.1

stopl = set(stopwords.words('english'))
stemmer = PorterStemmer()

#if sys.stdout.encoding is None:
#	os.putenv("PYTHONIOENCODING", 'UTF-8')
#	os.execv(sys.executable, ['python'] + sys.argv)


class TextFragments(object):

	def __iter__(self):
		for q in self.data.itervalues():
			for item in q.itervalues():
				yield item

	def calc_vectors(self):
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				rec["sent_vec"], rec["wv_tokens"] = wordvec.get_sentvec(rec["tokens"])

	def text(self):
		res = []
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				res.append(rec["tokens"])
		return res

	def wv_text(self):
		#res = []
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				#res.append(rec["wv_tokens"])
				for word in rec["wv_tokens"]:
					yield word
		#return res

	def wv_sentences(self):
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				yield rec["wv_tokens"]

	def wv_vocab(self):
		res = Counter()
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				res.update(rec["wv_tokens"])
		return res

	def normalize(self, matcher, df):
		printd("Normalizing dset")
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				rec["sent_vec"], rec["sent_sum"] = matcher.normalize(rec["sent_vec"], df)

	def vectorize(self, wordvec):
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				rec["sent_vec"], rec["wv_tokens"] = wordvec.get_sentvec(rec["tokens"])


class Superset(object):
	def __init__(self, *args):
		self.items = args

	def __len__(self):
		return sum([len(x) for x in self.items])

	def __iter__(self):
		return self.items.__iter__()


class NuggetDataset(Dataset):
	def __init__(self, nuggets, updates):
		self.nuggets = nuggets
		self.updates = updates
		self.data = Superset(nuggets, updates)

	def test(self):
		# Add labels if we need them from a matches class
		for nugget in self.nuggets:
			for update in self.updates:
				yield SentencePair(nugget, update)


class Nuggets(TextFragments):

	def __init__(self, filen, vectorize=False):
		self.nuggets = defaultdict(dict)
		self.vectorizep = vectorize
		self.read(filen)
		self.data = self.nuggets

	def read(self, filen):
		for rec in self.nuggetReader(filen):
			toks = tokenize(rec["text"])
			rec["tokens"] = stem(toks)
			if self.vectorizep:
				rec["sent_vec"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
				#rec["vec_sum"] = np.sum(rec["sent_vec"], axis=0)
			self.nuggets[rec["query_id"]][rec["id"]] = rec

	def nuggetReader(self, filen):
		with open(filen) as nh:
			for rec in UnicodeDictReader(nh, delimiter="\t", quoting=csv.QUOTE_NONE):
				rec["text"] = rec["nugget_text"]
				rec["id"] = rec["nugget_id"]
				yield(rec)


class TSNuggets(Nuggets):

	def nuggetReader(self, filen):
		with open(filen) as nh:
			for rec in UnicodeDictReader(nh, delimiter="\t", quoting=csv.QUOTE_NONE):
				rec["text"] = rec["nugget_text"]
				rec["id"] = rec["nugget_id"]
				yield(rec)


class CLNuggets(Nuggets):

	def nuggetReader(self, filen):
		vsfields = ["nugget_id", "impt", "length", "dep", "nugget_text"]
		for nf in glob.glob(filen):
			qid = "1C2-E-" + os.path.basename(nf).replace(".vitalstrings.txt", "")
			with open(nf) as nh:
				for rec in UnicodeDictReader(nh, fieldnames=vsfields, delimiter="\t",
																			quoting=csv.QUOTE_NONE):
					rec["query_id"] = qid
					rec["text"] = rec["nugget_text"]
					rec["id"] = rec["nugget_id"]
					yield(rec)


class MCNuggets(Nuggets):

	def nuggetReader(self, filen):
		with open(filen) as nh:
			for rec in UnicodeDictReader(nh, delimiter="\t", quoting=csv.QUOTE_NONE):
				rec["text"] = rec["vs_text"]
				rec["id"] = rec["vs_id"]
				yield(rec)


class Updates(TextFragments):

	def __init__(self, filen, vectorize=False):
		self.updates = defaultdict(dict)
		self.vectorizep = vectorize
		self.read(filen)
		self.data = self.updates

	def read(self, filen):
		for rec in self.updateReader(filen):
			if rec["duplicate_id"] != "NULL":
				continue
			toks = tokenize(rec["text"])
			rec["tokens"] = stem(toks)
			if self.vectorizep:
				rec["sent_vec"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
				#rec["vec_sum"] = np.sum(rec["sent_vec"], axis=0)
			self.updates[rec["query_id"]][rec["id"]] = rec

	def updateReader(self, filen):
		with open(filen) as nh:
			for rec in UnicodeDictReader(nh, delimiter="\t", quoting=csv.QUOTE_NONE):
				rec["text"] = rec["update_text"]
				rec["id"] = rec["update_id"]
				yield(rec)


class CLUpdates(Updates):
	def updateReader(self, filen):
		ufields = ["query_id", "type", "update_text"]
		for uf in glob.glob(filen):
			update_id = os.path.basename(uf).replace(".tsv", "")
			with open(uf) as uh:
				uh.readline()  # sysdesc
				for rec in UnicodeDictReader(uh, fieldnames=ufields, delimiter="\t",
																			quoting=csv.QUOTE_NONE):
					if rec["type"] != "OUT":
						continue
					rec["update_id"] = update_id
					rec["id"] = update_id
					rec["text"] = rec["update_text"]
					rec["duplicate_id"] = "NULL"
					yield(rec)


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

	def write(self, pair, match):
		nugget = pair.s1
		update = pair.s2
		qid = nugget["query_id"]
		print >>self.sh, "\t".join((qid, update["id"], nugget["id"],
																str(match.start), str(match.end),
																str(match.autop), "%g" % match.score,
																update["text"], nugget["text"]))


class CLMatchWriter(MatchWriter):

	def __init__(self, sf):
		self.sh = None
		self.qid = None
		self.sd = sf
		if sf and not os.path.exists(sf):
			os.makedirs(sf)

	def writeheader(self):
		pass

	def write(self, pair, match):
		nugget = pair.s1
		update = pair.s2
		qid = nugget["query_id"]
		if qid != self.qid:
			if self.sh:
				self.sh.__exit__(None, None, None)
			if self.sd:
				sf = os.path.join(self.sd, "%s.matches.txt" % (qid.replace("1C2-E-", "")))
			else:
				sf = os.devnull
			self.sh = codecs.open(sf, 'w', 'utf-8')
			self.qid = qid
		print >>self.sh, "\t".join((update["id"], nugget["id"],
																str(match.start), str(match.end),
																"%g" % match.score,
																update["text"], nugget["text"]))


class MCMatchWriter(MatchWriter):

	def writeheader(self):
		print >>self.sh, "\t".join(("query_id", "vs_id", "update_id",
																"update_source", "vs_start", "vs_end",
																"score", "update_text", "vs_text"))

	def write(self, pair, match):
		nugget = pair.s1
		update = pair.s2
		qid = nugget["query_id"]
		print >>self.sh, "\t".join((qid, nugget["id"], update["id"],
																update["update_source"],
																str(match.start), str(match.end),
																"%g" % match.score,
																update["text"], nugget["text"]))


def load_wordvec():
	return Word2Vec.load_word2vec_format(wordvecf, binary=True)


def UnicodeDictReader(utf8_data, **kwargs):
	csv_reader = csv.DictReader(utf8_data, **kwargs)
	for row in csv_reader:
		yield {key: unicode(value, 'utf-8') for key, value in row.iteritems()}


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


def tokenize(t):
	# separate numbers from other text
	# TODO(mattea): should account for sci-not (1e3)
	t = re.sub(r"([^\s\d])(?=\d)", r"\1 ", t)
	t = re.sub(r"(\d)(?=[^\s\d])", r"\1 ", t)
	try:
		toks = word_tokenize(t)
	except Exception:
		toks = t.strip().split()

	# Remove nonword characters and lowercase everything
	toks = [re.sub(r"[^\w\.\,\-]", "", x.lower()) for x in toks]
	return [x for x in toks if x not in stopl]


def stem(ts):
	return [stemmer.stem(x) for x in ts]


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


def vec_sim(s1, s2):
	return 1 - distance.cosine(s1["vec_sum"], s2["vec_sum"]), None
	#return distance.euclidean(vec1, vec2)


def min_sim(s1, s2):
	md = MinSim(s1["sent_vec"], s2["sent_vec"])
	return md.match(), md


def printd(string):
	if debug:
		print >> sys.stderr, string


#def main(nf, uf, sf, vf):
def main(args):
	global debug, wordvec, wordvecf
	if args.debug:
		debug = True
	nf = args.nuggets
	uf = args.updates
	sf = args.shingles
	vf = args.wordvec
	ef = args.evalfile
	wvout = args.wvfile
	sim_thr = args.sim_thr
	dset = args.dataset
	limit = args.limit

	if args.dataset == "auto":
		if ef is not None:
			dset = "semeval"
		else:
			with open(glob.glob(nf)[0]) as nh:
				nfhead = nh.readline()
				if nfhead.startswith("query_id\tnugget_id"):
					dset = "ts"
				elif nfhead.startswith("query_id\tvs_id"):
					dset = "mclick"
				else:
					dset = "1click"

	if os.path.exists(wvout) and not args.force:
		wordvecf = wvout

	if vf:
		printd("Reading word vector...")
		#wordvec = load_wordvec()
		wordvec = WordVec(wordvecf)
		if args.sim == "minsim":
			vecfn = MinSim
			wordvec.normalize()
		elif args.sim == "infsim":
			vecfn = InfSim
			wordvec.normalize()
		else:
			vecfn = VecSim

	if dset == "ts":
		nuggfn = Nuggets
		updfn = Updates
		outfn = MatchWriter
	elif dset == "1click":
		nuggfn = CLNuggets
		updfn = CLUpdates
		outfn = CLMatchWriter
	elif dset == "mclick":
		nuggfn = MCNuggets
		updfn = Updates
		outfn = MCMatchWriter
	elif dset == "semeval":
		data = SemEvalDataset(args.input_data, args.evalfile)
		outfn = data.writer
		if vf is not None:
			data.vectorize(wordvec)
	else:
		nuggfn = MCNuggets
		updfn = Updates
		outfn = MCMatchWriter

	if dset != "semeval":
		printd("Processing Nuggets...")
		nuggets = nuggfn(nf, vectorize=vf is not None)

		printd("Processing Updates...")
		updates = updfn(uf, vectorize=vf is not None)
		data = NuggetDataset(nuggets, updates)

	if vf and wvout is not None and wvout != wordvecf:
		printd("Rereading word vectors to optimize...")
		wv_toks = data.wv_sentences()
		#if dset == "semeval":
		#	wv_toks = data.wv_sentences()
		#else:
		#	wv_toks = nuggets.wv_text() + updates.wv_text()
		wordvec = WordVec(wordvecf, sentences=wv_toks, wvout=wvout, size=wordvec.originalsize)
		if args.sim == "infsim":
			wordvec.normalize()
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
	if not os.path.exists(logdffile) or (os.path.getmtime(logdffile) < os.path.getmtime(wordvecf)):
		np.savetxt(logdffile, logdf, delimiter=" ", fmt="%g")
	data.normalize(vecfn, logdf)

	printd("Finding matches...")
	with outfn(sf) as sw, outfn(vf) as vw:
		mcnt = 0
		timer = Timer()
		for pair in data.test():
			if sf:
				match = shingle(pair.s1["tokens"], pair.s2["tokens"])
				if match.score >= min_score:
					sw.write(pair, match)

			if vf:
				md = None
				try:
					md = vecfn(pair, logdf)
					sim = md.match()
					#printd("Match %0.4f for %s, %s" % (sim, nid, uid))
				except ValueError:
					sim = sim_thr
				if sim < sim_thr:
					sim = sim_thr
				if md:
					start = md.nmatches
					end = md.end - md.start
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
	argparser.add_argument('--sim_thr', type=float, help='Similarity threshold to use')
	argparser.add_argument('--limit', type=int, help='Limit number of matches performed (for testing)')
	main(argparser.parse_args(remaining_argv))
