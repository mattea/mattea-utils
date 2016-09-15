#!/usr/bin/python
from __future__ import division
import argparse
import ConfigParser
import codecs
import sys
import os
import traceback
#from itertools import chain
import numpy as np
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from itertools import izip
#from gensim.models import Word2Vec
#from scipy.spatial import distance
from matteautils.match.vectorsim import MinDistSim, InfSim, VecSim
from matteautils.parser import WordVec
#from matteautils.dataset.semeval import SemEvalDataset
import matteautils.dataset as dataset
from matteautils.base import printd
import matteautils.config as conf

sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def wvSum(p):
	return np.sum(p.s1["vector"], axis=0), np.sum(p.s2["vector"], axis=0)


def wvAvg(p):
	return np.average(p.s1["vector"], axis=0), np.average(p.s2["vector"], axis=0)


def wvMin(p):
	return np.amin(p.s1["vector"], axis=0), np.amin(p.s2["vector"], axis=0)


def wvMax(p):
	return np.amax(p.s1["vector"], axis=0), np.amax(p.s2["vector"], axis=0)


def wvCos(a):
	return [distance.cosine(x[0], x[1]) for x in a]


def wvDiff(a):
	res = []
	for x in a:
		res.extend(x[1] - x[0])
	return res


def wvIdent(a):
	#res = []
	#for x in a:
	#	res.extend(np.concatenate((x[0], x[1])))
	#return res
	return np.concatenate((a[0][0], a[0][1]))


def wvMaxDiff(a):
	return [np.amax(x[1] - x[0]) for x in a]


def wvMinDiff(a):
	return [np.amin(x[1] - x[0]) for x in a]


def wvEuc(a):
	return [distance.euclidean(x[0], x[1]) for x in a]


def wvMkow(a):
	return [distance.minkowski(x[0], x[1], 2) for x in a]


def formatFeature(f):
	if isinstance(f, np.ndarray):
		return "\t".join(["%g" % x for x in f])
	else:
		print "%g" % f


#featurefns = [wvSum, wvMin, wvMax, wvAvg, wvCos, wvDiff, wvEuc]
#featurefns = [wvCos, wvEuc, wvMkow]
featurefns = [wvCos, wvEuc, wvMkow, wvMaxDiff, wvMinDiff]
#featurefns = [wvCos, wvEuc, wvMkow, wvMaxDiff, wvMinDiff, wvIdent]
accumulators = [wvSum, wvMin, wvMax, wvAvg]


def featurize(p):
	a = []
	for accumulator in accumulators:
		a.append(accumulator(p))

	x = []
	for featurefn in featurefns:
		try:
			x.extend(featurefn(a))
		except Exception:
			print >>sys.stderr, "Pair:", p.pid
			print >>sys.stderr, "%d" % len(p.s1["vector"]), " ".join(p.s1["wv_tokens"])
			print >>sys.stderr, "%d" % len(p.s2["vector"]), " ".join(p.s2["wv_tokens"])
			traceback.print_exc()
			sys.exit(-1)
	return x


class FeatureSet(object):
	def __init__(self, data):
		self.train = []
		self.test = []
		self.trainlabels = []
		self.testlabels = []
		self.data = data

		for pair in data.train():
			self.trainlabels.append(pair.label)
			self.train.append(featurize(pair))
		for pair in data.test():
			self.testlabels.append(pair.label)
			self.test.append(featurize(pair))

		#self.trainlabels = np.array(self.trainlabels, dtype=np.float32)
		#self.testlabels = np.array(self.testlabels, dtype=np.float32)

	def addMatcher(self, matcher):
		for datum, pair in izip(self.train, self.data.train()):
			try:
				sim = matcher.match(pair)
			except ValueError, err:
				printd(err)
			datum.append(sim)

		for datum, pair in izip(self.test, self.data.test()):
			try:
				sim = matcher.match(pair)
			except ValueError, err:
				printd(err)
			datum.append(sim)

	def write(self, outf):
		with open(outf + ".train", 'w') as fh:
			for label, x in izip(self.trainlabels, self.train):
				print >>fh, "%g\t" % label,
				print >>fh, "\t".join(["%g" % v for v in x])

		with open(outf + ".test", 'w') as fh:
			for label, x in izip(self.testlabels, self.test):
				print >>fh, "%g\t" % label,
				print >>fh, "\t".join(["%g" % v for v in x])

	def freeze(self):
		self.train = np.asarray(self.train, dtype=np.float32)
		self.test = np.asarray(self.test, dtype=np.float32)
		self.trainlabels = np.asarray(self.trainlabels, dtype=np.float32)
		self.testlabels = np.asarray(self.testlabels, dtype=np.float32)


def evalData(model, data, labels):
	obs = model.predict(data)
	exp = labels

	pearsonsR, pearsonsP = pearsonr(obs, exp)
	spearmansR, spearmansP = spearmanr(obs, exp)
	ktau, ktauP = kendalltau(obs, exp)
	mse = mean_squared_error(exp, obs)
	print "Pearson's: %g (%g)" % (pearsonsR, pearsonsP)
	print "Spearman's: %g (%g)" % (spearmansR, spearmansP)
	print "Kendall's: %g (%g)" % (ktau, ktauP)
	print "MSE: %g" % (mse)


def main(args):
	wordvecf = '/huge1/sourcedata/word2vec/google/GoogleNews-vectors-negative300.bin'
	conf.debug = args.debug
	conf.verbose = args.verbose
	if args.alphas == "auto":
		pass  # TODO
	else:
		conf.alphas = [float(x) for x in args.alphas.split(",")]
	data = dataset.Dataset.load(args.input_data, args.dataset)
	#data = SemEvalDataset(args.input_data)

	wvout = args.wvfile
	if os.path.exists(wvout):
		wordvecf = wvout
	wordvec = WordVec(wordvecf)
	data.vectorize(wordvec)

	if wvout != wordvecf:
		printd("Rereading word vectors to optimize...")
		wv_toks = data.wv_sentences()
		wordvec = WordVec(wordvecf, sentences=wv_toks, wvout=wvout, size=wordvec.originalsize)
		data.vectorize(wordvec)

	try:
		max_features = int(args.max_features)
	except ValueError:
		try:
			max_features = float(args.max_features)
		except ValueError:
			max_features = args.max_features

	# Train data
	fs = FeatureSet(data)
	#trainlabels = []
	#traindata = []
	#for pair in data.train():
	#	trainlabels.append(pair.label)
	#	traindata.append(featurize(pair))

	# Test data
	#testlabels = []
	#testdata = []
	#for pair in data.test():
	#	testdata.append(featurize(pair))
	#	testlabels.append(pair.label)

	comparator = 'cosine'
	matcher = MinDistSim(metric=comparator)
	fs.addMatcher(matcher)

	# We normalize after so primary features are raw word vectors
	#wordvec.normalize()
	# InfSim
	wordvec.normalize()
	logdf = wordvec.logdf(data.wv_vocab())
	matcher = InfSim(df=logdf)
	data.vectorize(wordvec)
	data.normalize(matcher, logdf)
	fs.addMatcher(matcher)

	fs.freeze()

	if (not args.force) and args.model and os.path.exists(args.model):
		model = joblib.load(args.model)
	else:
		if args.features and (args.force or not os.path.exists(args.features)):
			fs.write(args.features)
		#dtree = DecisionTreeRegressor()
		#kpca = KernelPCA(kernel="rbf", gamma=10)
		model = RandomForestRegressor(max_depth=args.max_depth, n_estimators=args.max_trees, oob_score=True, max_features=max_features, n_jobs=-1)
		#model = Pipeline(steps=[('pca', kpca), ('dtree', dtree)])
		printd("Running Pipeline")
		model.fit(fs.train, fs.trainlabels)
		#X_kpca = kpca.fit_transform(X)
		#dtree.fit(traindata, trainlabels)
		if args.model:
			joblib.dump(model, args.model)

	printd("Evaluating")
	# train accuracy
	print "Train Accuracy"
	evalData(model, fs.train, fs.trainlabels)

	# test accuracy
	print "Test Accuracy"
	evalData(model, fs.test, fs.testlabels)


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
		"max_depth": 4,
		"max_trees": 40,
		"alphas": "auto",
		"max_features": "auto",
		"limit": None,
		}
	if args.conf_file:
		config = ConfigParser.SafeConfigParser()
		config.read([args.conf_file])
		defaults = dict(config.items("Defaults"))

	argparser = argparse.ArgumentParser(description='Automatically match using shingles and word vectors', parents=[conf_parser])
	argparser.set_defaults(**defaults)
	argparser.add_argument('-i', '--input_data', help='Dataset file/dir')
	argparser.add_argument('--dataset', help='The name of the dataset (otherwise inferred from files)')
	argparser.add_argument('-d', '--debug', action='store_true', help='Debug mode (lots of output)')
	argparser.add_argument('--verbose', action='store_true', help='Even more output')
	argparser.add_argument('-v', '--wordvec', help='Write out wordvec to file/dir')
	argparser.add_argument('--model', help='Write out model to file, restore if exists')
	argparser.add_argument('--features', help='Write out features to file, if does exists')
	argparser.add_argument('--frequencies', help='Read from word frequencies JSON file')
	argparser.add_argument('-f', '--force', action='store_true', help='Relearning and rewriting files')
	argparser.add_argument('--max_depth', type=int, help='Max depth of a tree')
	argparser.add_argument('--alphas', help='Weight of each wordvec')
	argparser.add_argument('--max_trees', type=int, help='Max number of trees (for random forests)')
	argparser.add_argument('--max_features', help='Max number of features to consider for split (for random forests)')
	argparser.add_argument('-w', '--wvfile', help='Store minimal version of wordvec file after processing, ' +
			'for faster subsequent runs (reused if exists, unless -f used).')
	argparser.add_argument('--learn', help='Write features for a learning algorithm')
	main(argparser.parse_args(remaining_argv))
