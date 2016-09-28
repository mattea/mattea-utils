class Matcher(object):
	def __init__(self, df=None, metric=None):
		pass

	def match(self, pair):
		self.features = []
		self.tsim = None

	@classmethod
	def normalize(cls, s, df):
		return s, 1

	def features(self):
		try:
			return self._features
		except AttributeError:
			return [self.tsim]

	def names(self):
		try:
			return self._names
		except AttributeError:
			return [self.__class__.__name__]

	def __unicode__(self):
		return "\n".join((self.pair.s1["text"],
										self.pair.s2["text"],
										" ".join(self.pair.s1["wv_tokens"]),
										" ".join(self.pair.s2["wv_tokens"]),
										"Score: %g" % self.tsim,
										"Label: %g" % self.pair.label))


class Match(object):
	def __init__(self, score=0.0, start=0, end=0, autop=1):
		self.score = score
		self.start = start
		self.end = end
		self.autop = autop
