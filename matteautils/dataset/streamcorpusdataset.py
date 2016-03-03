import dataset
import streamcorpus as sc
from matteautils.base import *

class StreamCorpusDataset(Dataset):
	def __init__(self, filenames, **kwargs):
		super(StreamCorpusDataset, self).__init__(kwargs)

		filenames = getfiles(filenames)

		for filename in filenames:
			for si in sc.Chunk(path=filename):
				if si.body.clean_visible == None:
					continue

				did = si.stream_id

				try:
					sentences = si.body.sentences["serif"]
				except KeyError:
					sentences = si.body.sentences["lingpipe"]

				for sind, sentence in enumerate(sentences):
					sid = make_sid(did, sind)
					self.add_sentence(sid, sentence)

		self.build_dictionary()


def make_sid(did, sind):
	return "%s-%s" % (did, sind)
