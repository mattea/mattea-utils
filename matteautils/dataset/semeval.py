from .. import TSVReader
from ..parser import tokenize
import sys
import os
import dataset
from dataset import Dataset, SentencePair, MatchWriter


def printd(s):
	print >>sys.stderr, s


#pair_ID sentence_A  sentence_B  relatedness_score entailment_judgment
class SemEvalDataset(Dataset):

	def __init__(self, path, **kwargs):
		super(SemEvalDataset, self).__init__(**kwargs)

		train_file = os.path.join(path, "SICK_train.txt")
		#sts_train_file = os.path.join(path, "STS_train.txt")
		test_file = os.path.join(path, "SICK_test_annotated.txt")
		valid_file = os.path.join(path, "SICK_trial.txt")

		ntrain = self.readFile(train_file)
		#if os.path.exists(sts_train_file):
		#	ntrain += self.readFile(sts_train_file)
		nvalid = self.readFile(valid_file)
		self.readFile(test_file)

		#self.build_dictionary()
		#self.index_data()
		#self.train = ROListSlice(self.data, 0, ntrain)
		#self.test = ROListSlice(self.data, ntrain, len(self.data))

		ntrain += nvalid

		self._train = self.data[:ntrain]
		self.trainlabels = self.labels[:ntrain]

		#self._valid = self.data[ntrain:ntrain + nvalid]
		#self.validlabels = self.labels[ntrain:ntrain + nvalid]

		#self._test = self.data[ntrain + nvalid:]
		#self.testlabels = self.labels[ntrain + nvalid:]

		self._test = self.data[ntrain:]
		self.testlabels = self.labels[ntrain:]

	# Do we want to try to grab entailment and learn models for it as a later
	# feature?
	def readFile(self, filen):
		nlines = 0
		for line in TSVReader(filen):
			score = float(line["relatedness_score"])
			#self.add_tokens(make_pair(line["sentence_A"], line["sentence_B"]))
			self.add_tokens(SentencePair(line["sentence_A"], line["sentence_B"], pid=line["pair_ID"], label=score))
			self.add_label(score)
			nlines += 1

		return nlines

	@classmethod
	def identify(cls, path):
		pfiles = os.listdir(path)
		return "SICK_train.txt" in pfiles

	@classmethod
	def writer(cls, *args, **kwargs):
		return SemEvalWriter(*args, **kwargs)

dataset.dtypes["semeval"] = SemEvalDataset


def make_pair(s1, s2):
		return tokenize(s1) + ["<s1e>"] + tokenize(s2)


class SemEvalWriter(MatchWriter):

	def writeheader(self):
		print >>self.sh, "\t".join(("pair_ID", "sentence_A", "sentence_B", "relatedness_score", "entailment_judgment", "gold_score"))

	def write(self, pair, match):
		print >>self.sh, "\t".join((pair.pid, pair.s1["text"], pair.s2["text"],
																"%0.4f" % match.score, "", "%0.4f" % pair.label))
