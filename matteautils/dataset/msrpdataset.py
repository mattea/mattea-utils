from itertools import chain
from . import Dataset
from .. import TSVReader
from nltk import word_tokenize

class MSRPDataset(Dataset):

	def __init__(self, train_file, test_file, **kwargs):
		super(MSRPDataset, self).__init__(**kwargs)

		ntrain = 0
		for line in TSVReader(train_file):
			#self.add_tokens(SentencePair(line["#1 String"], line["#2 String"]))
			self.add_tokens(make_pair(line["#1 String"], line["#2 String"]))
			self.add_label(int(line["Quality"]))
			ntrain += 1

		ntest = 0
		for line in TSVReader(test_file):
			#self.add_tokens(SentencePair(line["#1 String"], line["#2 String"]))
			self.add_tokens(make_pair(line["#1 String"], line["#2 String"]))
			self.add_label(int(line["Quality"]))
			ntest += 1

		self.build_dictionary()
		self.index_data()
		#self.train = ROListSlice(self.data, 0, ntrain)
		#self.test = ROListSlice(self.data, ntrain, len(self.data))
		self.train = self.data[:ntrain]
		self.trainlabels = self.labels[:ntrain]
		self.test = self.data[ntrain:]
		self.testlabels = self.labels[ntrain:]


def make_pair(s1, s2):
		return word_tokenize(s1) + ["<s1e>"] + word_tokenize(s2)


class SentencePair(object):

	def __init__(self, s1, s2, sid1, sid2):
		if type(s1) == str:
			s1 = tokenize(s1)
		if type(s2) == str:
			s2 = tokenize(s2)
		self.s1 = s1
		self.s2 = s2
		self.sid1 = sid1
		self.sid2 = sid2
		self.__len = len(s1) + len(s2)

	def __str__(self):
		return " ".join(self.s1) + " <s1e> " + " ".join(self.s2)

	def __iter__(self):
		for word in chain(self.s1, self.s2):
			yield word

	def __getitem__(self, index):
		s1l = len(s1)
		if s1l > index:
			return s1[index]
		else:
			return s2[index - s1l]

	def __setitem__(self, index, val):
		s1l = len(s1)
		if s1l > index:
			s1[index] = val
		else:
			s2[index - s1l] = val

	def __len__(self):
		return self.__len

