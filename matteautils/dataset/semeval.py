from itertools import chain
from . import Dataset, MatchWriter
from .. import TSVReader
from ..parser import tokenize
import sys


def printd(s):
	print >>sys.stderr, s


#pair_ID sentence_A  sentence_B  relatedness_score entailment_judgment
class SemEvalDataset(Dataset):

	def __init__(self, train_file, test_file, **kwargs):
		super(SemEvalDataset, self).__init__(**kwargs)

		ntrain = 0
		for line in TSVReader(train_file):
			score = float(line["relatedness_score"])
			#self.add_tokens(make_pair(line["sentence_A"], line["sentence_B"]))
			self.add_tokens(SentencePair(line["sentence_A"], line["sentence_B"], pid=line["pair_ID"], label=score))
			self.add_label(score)
			ntrain += 1

		ntest = 0
		for line in TSVReader(test_file):
			score = float(line["relatedness_score"])
			#self.add_tokens(make_pair(line["sentence_A"], line["sentence_B"]))
			self.add_tokens(SentencePair(line["sentence_A"], line["sentence_B"], pid=line["pair_ID"], label=score))
			self.add_label(score)
			ntest += 1

		#self.build_dictionary()
		#self.index_data()
		#self.train = ROListSlice(self.data, 0, ntrain)
		#self.test = ROListSlice(self.data, ntrain, len(self.data))
		self._train = self.data[:ntrain]
		self.trainlabels = self.labels[:ntrain]
		self._test = self.data[ntrain:]
		self.testlabels = self.labels[ntrain:]

	def writer(self, *args, **kwargs):
		return SemEvalWriter(*args, **kwargs)


def make_pair(s1, s2):
		return tokenize(s1) + ["<s1e>"] + tokenize(s2)


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
		return " ".join(self.s1) + " <s1e> " + " ".join(self.s2)

	def __iter__(self):
		for word in chain(self.s1["tokens"], self.s2["tokens"]):
			yield word

	def wv_text(self):
		for word in chain(self.s1["wv_tokens"], self.s2["wv_tokens"]):
			yield word

	def wv_sentences(self):
		return [self.s1["wv_tokens"], self.s2["wv_tokens"]]

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


class SemEvalWriter(MatchWriter):

	def writeheader(self):
		print >>self.sh, "\t".join(("pair_ID", "sentence_A", "sentence_B", "relatedness_score", "entailment_judgment", "gold_score"))

	def write(self, pair, match):
		print >>self.sh, "\t".join((pair.pid, pair.s1["text"], pair.s2["text"],
																"%0.4f" % match.score, "", "%0.4f" % pair.label))
