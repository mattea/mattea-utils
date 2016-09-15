import csv
import glob
import os
import codecs
import re
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from matteautils.base import TSVReader, UnicodeDictReader, printd
import dataset
from dataset import Dataset, SentencePair, MatchWriter

stopl = set(stopwords.words('english'))
stemmer = PorterStemmer()


class NuggetDataset(Dataset):
	def __init__(self, path):
		pfiles = os.listdir(path)
		printd("Loading dataset")
		alldata = []
		if "train" in pfiles:
			for d in ["train", "test", "valid"]:
				if d not in pfiles:
					continue
				dpath = os.path.join(path, d)
				nset = NuggetSet(dpath)
				alldata.extend([nset.nuggets, nset.updates])
				setattr(self, d, nset.pairs)
		else:
			nset = NuggetSet(path)
			alldata.extend([nset.nuggets, nset.updates])
			self.test = nset.pairs

		self.data = Superset(*alldata)

	@classmethod
	def identify(cls, path):
		pfiles = os.listdir(path)
		if "train" in pfiles:
			pfiles += os.listdir(os.path.join(path, "train"))
		return "nuggets.tsv" in pfiles or "gold_iunits.tsv" in pfiles or "0001.matches.txt" in pfiles

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

dataset.dtypes["ts"] = NuggetDataset


class NuggetSet(object):
	def __init__(self, path):
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

	def pairs(self):
		matches = self.matches
		for nugget in self.nuggets:
			for update in self.updates:
				yield SentencePair(nugget, update, label=matches.match(nugget, update))


class TextFragments(object):

	def __iter__(self):
		for q in self.data.itervalues():
			for item in q.itervalues():
				yield item

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
				rec["vector"], rec["vector_sum"] = matcher.normalize(rec["vector"], df)

	def vectorize(self, wordvec):
		for qid, rs in self.data.iteritems():
			for rid, rec in rs.iteritems():
				rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(rec["tokens"])


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
			#if self.vectorizep:
			#	rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
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
			#if self.vectorizep:
			#	rec["vector"], rec["wv_tokens"] = wordvec.get_sentvec(toks)
				#rec["vec_sum"] = np.sum(rec["vector"], axis=0)
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


class Matches(object):
	def __init__(self, filen):
		matches = dict()
		self.matches = matches
		if filen is None:
			return
		for rec in TSVReader(filen):
			matches[rec["nugget_id"] + rec["update_id"]] = rec

	def __getitem__(self, key):
		return self.matches[key]

	def match(self, nid, uid):
		if not isinstance(nid, str):
			nid = nid["id"]
		if not isinstance(uid, str):
			uid = uid["id"]
		return 1 if nid + uid in self.matches else 0


class CLMatches(Matches):
	def __init__(self, files):
		matches = dict()
		self.matches = matches
		mfields = ["update_id", "nugget_id", "start", "end"]
		if files is None:
			return
		for filen in files:
			for rec in TSVReader(filen, fieldnames=mfields):
				matches[rec["nugget_id"] + rec["update_id"]] = rec


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
	def __init__(self, *args):
		self.items = args

	def __len__(self):
		return sum([len(x) for x in self.items])

	def __iter__(self):
		return self.items.__iter__()


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
