import csv
import glob
import os
import codecs
import re
import sys
import traceback
#import random
from collections import Counter
#from itertools import chain
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from matteautils.randomdict import RandomDict
from matteautils.base import TSVReader, UnicodeDictReader, printd
import dataset
from dataset import Dataset, SentencePair, MatchWriter
import matteautils.config as conf

try:
	stopl = set(stopwords.words('english'))
except LookupError:
	nltk.download('stopwords')
	nltk.download('ptb')
	nltk.download('punkt')
	stopl = set(stopwords.words('english'))
stemmer = PorterStemmer()


class NuggetDataset(Dataset):
	def __init__(self, path, neg_samples=5):
		pfiles = os.listdir(path)
		printd("Loading dataset")
		alldata = []
		if "train" in pfiles:
			for d in ["train", "test", "valid"]:
				if d not in pfiles:
					continue
				dpath = os.path.join(path, d)
				nset = NuggetSet(dpath, neg_samples)
				alldata.extend([nset.nuggets, nset.updates])
				setattr(self, d, nset.pairs)
		else:
			nset = NuggetSet(path, neg_samples)
			alldata.extend([nset.nuggets, nset.updates])
			self.test = nset.pairs

		self.data = Superset(*alldata)
		self.writer = nset.writer

	@classmethod
	def identify(cls, path):
		pfiles = os.listdir(path)
		if "train" in pfiles:
			pfiles += os.listdir(os.path.join(path, "train"))

		return (("nuggets.tsv" in pfiles) or
				("gold_iunits.tsv" in pfiles) or
				("0001.matches.txt" in pfiles))

	def train(self):
		yield None

	def valid(self):
		yield None

	def test(self):
		yield None
		#matches = self.matches
		#for nugget in self.nuggets:
		#	for update in self.updates:
		#		yield SentencePair(nugget, update, label=matches.match(nugget, update))

	def maxShortSentence(self):
		ls = Cycle([0, 0])

		try:
			for dset in self.data:
				l = ls.nextitem()
				for s in dset:
					cl = len(s["wv_tokens"])
					if cl > l:
						l = cl
				ls.setitem(l)
		except KeyError, e:
			printd(e, -1)
			printd(s, -1)
			traceback.print_stack()
			sys.exit(-1)
		return min(ls)

dataset.dtypes["ts"] = NuggetDataset


class Cycle(list):
	def nextitem(self):
		try:
			self._curritem = (self._curritem + 1) % len(self)
		except AttributeError:
			self._curritem = 0
		return self[self._curritem]

	def setitem(self, v):
		self[self._curritem] = v


class NuggetSet(object):
	def __init__(self, path, neg_samples=None):
		pfiles = os.listdir(path)
		if "nuggets.tsv" in pfiles:
			self.nuggets = Nuggets(os.path.join(path, "nuggets.tsv"))
			self.updates = Updates(os.path.join(path, "updates_sampled.tsv"))
			self.matches = Matches(os.path.join(path, "matches.tsv"))
			self.writer = MatchWriter
		elif "0001.matches.txt" in pfiles:
			self.nuggets = CLNuggets(glob.glob(os.path.join(path, "*.vitalstrings.txt")))
			self.updates = CLUpdates(glob.glob(os.path.join(path, "*.summaries.txt")))
			self.matches = CLMatches(glob.glob(os.path.join(path, "*.matches.txt")))
			self.writer = CLMatchWriter
		elif "gold_iunits.tsv" in pfiles:
			self.nuggets = MCNuggets(os.path.join(path, "gold_iunits.tsv"))
			self.updates = Updates(os.path.join(path, "pooled_iunits.tsv"))
			self.matches = Matches(os.path.join(path, "matches.tsv"))
			self.writer = MCMatchWriter

		self.p = len(self.matches) / (len(self.nuggets) * len(self.updates))
		if neg_samples is not None:
			self.neg_samples = neg_samples
		elif conf.neg_samples:
			self.neg_samples = conf.neg_samples
		else:
			self.neg_samples = 5

	# neg_samples is the number of negative samples to use for each positive
	# sample (possibly in expectation)
	def pairs(self):
		try:
			return self._pairs
		except AttributeError:
			pass

		pairs = []
		self._pairs = pairs
		matches = self.matches
		#p = self.p * neg_samples
		## Method assumes # matches ~ #upd*#nugg
		#for nugget in self.nuggets:
		#	for update in self.updates:
		#		match = matches.match(nugget, update)
		#		if match:
		#			yield SentencePair(nugget, update, label=match)
		#		elif random.random() < p:
		#			yield SentencePair(nugget, update, label=match)

		# Method assumes # matches << #upd*#nugg
		nuggets = self.nuggets
		updates = self.updates
		neg_samples = self.neg_samples
		for match in matches:
			nid = match["nugget_id"]
			uid = match["update_id"]
			if (nid not in nuggets) or (uid not in updates):
				continue
			pairs.append(SentencePair(nuggets[nid], updates[uid], label=1.0))

			for _ in range(neg_samples):
				while True:
					rnid, nugg = nuggets.random_item()
					ruid, upd = updates.random_item()
					matchp = matches.match(rnid, ruid)
					if not matchp:
						break
				pairs.append(SentencePair(nugg, upd, label=0.0))

		return pairs


class TextFragments(object):

	def __iter__(self):
		for item in self.data.itervalues():
			yield item

	def __getitem__(self, key):
		return self.data[key]

	def __contains__(self, key):
		return key in self.data

	def random_item(self):
		return self.data.random_item()

	def random_key(self):
		return self.data.random_key()

	def random_value(self):
		return self.data.random_value()

	def __len__(self):
		return self.count

	def text(self):
		res = []
		for rid, rec in self.data.iteritems():
			res.append(rec["tokens"])
		return res

	def wv_text(self):
		#res = []
		for rid, rec in self.data.iteritems():
			#res.append(rec["wv_tokens"])
			for word in rec["wv_tokens"]:
				yield word
		#return res

	def wv_sentences(self):
		for rid, rec in self.data.iteritems():
				yield rec["wv_tokens"]

	def wv_vocab(self):
		try:
			return self._wv_vocab
		except AttributeError:
			pass
		res = Counter()
		for rid, rec in self.data.iteritems():
				res.update(rec["wv_tokens"])
		self._wv_vocab = res
		return res

	def normalize(self, matcher, df):
		printd("Normalizing dset")
		for rid, rec in self.data.iteritems():
				rec["vector"], rec["vector_sum"] = matcher.normalize(rec["vector"], df)

	def vectorize(self, wordvec):
		for rid, rec in self.data.items():
				rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(rec["tokens"])
				if len(rec["vector"]) == 0:
					printd("Empty vector for:", 1)
					printd(rec, 1)
					del self.data[rid]


class Nuggets(TextFragments):

	def __init__(self, filen, vectorize=False):
		self.nuggets = RandomDict()
		self.vectorizep = vectorize
		self.read(filen)
		self.data = self.nuggets

	def read(self, filen):
		count = 0
		for rec in self.nuggetReader(filen):
			toks = tokenize(rec["text"])
			if len(toks) == 0:
				continue
			#rec["tokens"] = stem(toks)
			rec["tokens"] = toks
			#if self.vectorizep:
			#	rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
			self.nuggets[rec["id"]] = rec
			count += 1

		self.count = count

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
		self.updates = RandomDict()
		self.vectorizep = vectorize
		self.read(filen)
		self.data = self.updates

	def read(self, filen):
		count = 0
		for rec in self.updateReader(filen):
			if rec["duplicate_id"] != "NULL":
				continue
			toks = tokenize(rec["text"])
			if len(toks) == 0:
				continue
			#rec["tokens"] = stem(toks)
			rec["tokens"] = toks
			#if self.vectorizep:
			#	rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
				#rec["vec_sum"] = np.sum(rec["vector"], axis=0)
			self.updates[rec["id"]] = rec
			count += 1
		self.count = count

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


class Matches(object):
	def __init__(self, filen):
		matches = dict()
		self.matches = matches
		if filen is None:
			return

		count = 0
		for rec in self.reader(filen):
			matches[rec["nugget_id"] + rec["update_id"]] = rec
			count += 1

		self.count = count

	def reader(self, filen):
		for rec in TSVReader(filen):
			yield rec

	def __getitem__(self, key):
		return self.matches[key]

	def __contains__(self, key):
		return key in self.matches

	def __len__(self):
		return self.count

	def match(self, nid, uid):
		if not isinstance(nid, basestring):
			nid = nid["id"]
		if not isinstance(uid, basestring):
			uid = uid["id"]
		return 1 if nid + uid in self.matches else 0

	def __iter__(self):
		return iter(self.matches.itervalues())


class CLMatches(Matches):
	def reader(self, files):
		mfields = ["update_id", "nugget_id", "start", "end"]
		for filen in files:
			for rec in TSVReader(filen, fieldnames=mfields):
				yield rec


#class MatchWriter(object):
#
#	def __init__(self, sf):
#		self.sh = codecs.open(sf or os.devnull, 'w', 'utf-8')
#
#	def __enter__(self):
#		if self.sh:
#			self.sh.__enter__()
#			self.writeheader()
#		return self
#
#	def __exit__(self, ctype, value, traceback):
#		if self.sh:
#			self.sh.__exit__(ctype, value, traceback)
#
#	def writeheader(self):
#		print >>self.sh, "\t".join(("QueryID", "UpdateID", "NuggetID", "Start",
#																"End", "AutoP", "Score", "Update_Text",
#																"Nugget_Text"))
#
#	def write(self, pair, match):
#		nugget = pair.s1
#		update = pair.s2
#		qid = nugget["query_id"]
#		print >>self.sh, "\t".join((qid, update["id"], nugget["id"],
#																str(match.start), str(match.end),
#																str(match.autop), "%g" % match.score,
#																update["text"], nugget["text"]))


class Superset(object):
	def __init__(self, *items):
		self.items = items

	def __len__(self):
		return sum([len(x) for x in self.items])

	def __iter__(self):
		return self.items.__iter__()
		#for dset in self.items:
		#	for item in dset:
		#		yield item


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


def tokenize(t):
	# separate numbers from other text
	# TODO(mattea): should account for scientific-notation (e.g. 1e3)
	t = re.sub(r"([^\s\d])(?=\d)", r"\1 ", t)
	t = re.sub(r"(\d)(?=[^\s\d])", r"\1 ", t)
	try:
		toks = word_tokenize(t)
	except Exception:
		toks = t.strip().split()

	# Remove nonword characters and lowercase everything
	toks = [re.sub(r"[^\w\.\,\-]", "", x.lower()) for x in toks]
	#return [x for x in toks if x not in stopl]
	return toks


def stem(ts):
	return [stemmer.stem(x) for x in ts]
