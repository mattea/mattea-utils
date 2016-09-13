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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from itertools import izip
#from gensim.models import Word2Vec
#from scipy.spatial import distance
from matteautils.match.vectorsim import MinDistSim, InfSim, VecSim
from matteautils.parser import WordVec
from matteautils.dataset.semeval import SemEvalDataset
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
featurefns = [wvCos, wvEuc, wvMkow, wvMaxDiff, wvMinDiff, wvIdent]
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


def main(args):
	conf.debug = args.debug
	conf.verbose = args.verbose
	data = SemEvalDataset(args.input_data, args.evalfile)
	wordvec = WordVec(args.wvfile)
	data.vectorize(wordvec)

	try:
		max_features = int(args.max_features)
	except ValueError:
		try:
			max_features = float(args.max_features)
		except ValueError:
			max_features = args.max_features

	# Train data
	trainlabels = []
	traindata = []
	for pair in data.train():
		trainlabels.append(pair.label)
		traindata.append(featurize(pair))

	# Test data
	testlabels = []
	testdata = []
	for pair in data.test():
		testdata.append(featurize(pair))
		testlabels.append(pair.label)

	# We normalize after so primary features are raw word vectors
	#wordvec.normalize()
	comparator = 'cosine'
	matcher = MinDistSim(metric=comparator)

	for datum, pair in izip(traindata, data.train()):
		try:
			sim = matcher.match(pair)
		except ValueError, err:
			printd(err)
		datum.append(sim)

	for datum, pair in izip(testdata, data.test()):
		try:
			sim = matcher.match(pair)
		except ValueError, err:
			printd(err)
		datum.append(sim)

	if (not args.force) and args.model and os.path.exists(args.model):
		dtree = joblib.load(args.model)
	else:
		if args.features and (args.force or not os.path.exists(args.features)):
			with open(args.features) as fh:
				for label, x in izip(trainlabels, traindata):
					print >>fh, "%g\t" % label,
					for f in x:
						print >>fh, "".join(["%g\t" % v for v in f]),
		#dtree = DecisionTreeRegressor()
		dtree = RandomForestClassifier(max_depth=args.max_depth, n_estimators=args.max_trees, oob_score=True, max_features=max_features, n_jobs=-1)
		dtree.fit(traindata, trainlabels)
		if args.model:
			joblib.dump(dtree, args.model)

	# train accuracy
	obs = dtree.predict(traindata)
	exp = trainlabels

	pearsonsR, pearsonsP = pearsonr(obs, exp)
	spearmansR, spearmansP = spearmanr(obs, exp)
	ktau, ktauP = kendalltau(obs, exp)
	mse = mean_squared_error(exp, obs)
	print "Train Accuracy"
	print "Pearson's: %g (%g)" % (pearsonsR, pearsonsP)
	print "Spearman's: %g (%g)" % (spearmansR, spearmansP)
	print "Kendall's: %g (%g)" % (ktau, ktauP)
	print "MSE: %g" % (mse)

	# test accuracy
	obs = dtree.predict(testdata)
	exp = testlabels

	pearsonsR, pearsonsP = pearsonr(obs, exp)
	spearmansR, spearmansP = spearmanr(obs, exp)
	ktau, ktauP = kendalltau(obs, exp)
	mse = mean_squared_error(exp, obs)
	print "Test Accuracy"
	print "Pearson's: %g (%g)" % (pearsonsR, pearsonsP)
	print "Spearman's: %g (%g)" % (spearmansR, spearmansP)
	print "Kendall's: %g (%g)" % (ktau, ktauP)
	print "MSE: %g" % (mse)


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
	argparser.add_argument('-d', '--debug', action='store_true', help='Debug mode (lots of output)')
	argparser.add_argument('--verbose', action='store_true', help='Even more output')
	argparser.add_argument('-e', '--evalfile', help='Single Eval file (e.g. SICK)')
	argparser.add_argument('-v', '--wordvec', help='Write out wordvec to file/dir')
	argparser.add_argument('--model', help='Write out model to file, restore if exists')
	argparser.add_argument('--features', help='Write out features to file, if does exists')
	argparser.add_argument('--frequencies', help='Read from word frequencies JSON file')
	argparser.add_argument('-f', '--force', action='store_true', help='Relearning and rewriting files')
	argparser.add_argument('--max_depth', type=int, help='Max depth of a tree')
	argparser.add_argument('--max_trees', type=int, help='Max number of trees (for random forests)')
	argparser.add_argument('--max_features', help='Max number of features to consider for split (for random forests)')
	argparser.add_argument('-w', '--wvfile', help='Store minimal version of wordvec file after processing, ' +
			'for faster subsequent runs (reused if exists, unless -f used).')
	argparser.add_argument('--learn', help='Write features for a learning algorithm')
	main(argparser.parse_args(remaining_argv))
