#!/usr/bin/python
from __future__ import division
import argparse
import ConfigParser
import codecs
import sys
import os
import traceback
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import sklearn.ensemble as ensemble
#from sklearn.decomposition import KernelPCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#from autosklearn.classification import AutoSklearnClassifier
#from sklearn.pipeline import Pipeline
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from math import floor, ceil
import itertools
from itertools import izip, chain
#from gensim.models import Word2Vec
#from scipy.spatial import distance
import matteautils.match.vectorsim as vectorsim
from matteautils.parser import WordVec, Shingler
#from matteautils.dataset.semeval import SemEvalDataset
import matteautils.dataset as dataset
from matteautils.base import printd
import matteautils.config as conf
from . import Match

sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def pairSum(p):
	return np.sum(p.s1["vector"], axis=0), np.sum(p.s2["vector"], axis=0)


def pairAvg(p):
	return np.average(p.s1["vector"], axis=0), np.average(p.s2["vector"], axis=0)


def pairMin(p):
	return np.amin(p.s1["vector"], axis=0), np.amin(p.s2["vector"], axis=0)


def pairMax(p):
	return np.amax(p.s1["vector"], axis=0), np.amax(p.s2["vector"], axis=0)


def wvDot(a):
	return [np.dot(x[0], x[1]) for x in a]


def wvCos(a):
	return [distance.cosine(x[0], x[1]) for x in a]


def wvDiff(a):
	res = []
	for x in a:
		res.extend(abs(x[1] - x[0]))
	return res


def wvSum(a):
	res = []
	for x in a:
		res.extend(abs(x[1] + x[0]))
	return res


def wvMult(a):
	res = []
	for x in a:
		res.extend(x[1] * x[0])
	return res


def wvIdent(a):
	#res = []
	#for x in a:
	#	res.extend(np.concatenate((x[0], x[1])))
	#return res
	#Only doing sum for now
	return np.concatenate((a[0][0], a[0][1]))


def wvMaxDiff(a):
	return [np.amax(x[1] - x[0]) for x in a]


def wvMinDiff(a):
	return [np.amin(x[1] - x[0]) for x in a]


def wvEuc(a):
	return [distance.euclidean(x[0], x[1]) for x in a]


def wvCity(a):
	return [distance.cityblock(x[0], x[1]) for x in a]


def wvCheb(a):
	return [distance.chebyshev(x[0], x[1]) for x in a]


def wvCanb(a):
	return [distance.canberra(x[0], x[1]) for x in a]


def wvCorr(a):
	return [distance.correlation(x[0], x[1]) for x in a]


def wvBray(a):
	return [distance.braycurtis(x[0], x[1]) for x in a]


def wvMkow(a):
	return [distance.minkowski(x[0], x[1], 2) for x in a]


def wvStd(a):
	return [np.std(abs(x[0] - x[1])) for x in a]


def formatFeature(f):
	if isinstance(f, np.ndarray):
		return "\t".join(["%g" % x for x in f])
	else:
		print "%g" % f


#featurefns = [wvSum, wvMin, wvMax, wvAvg, wvCos, wvDiff, wvEuc]
#featurefns = [wvCos, wvEuc, wvMkow]
#featurefns = [wvCos, wvEuc, wvMkow, wvMaxDiff, wvMinDiff]
#featurefns = [wvCos, wvEuc, wvMkow, wvMaxDiff, wvMinDiff, wvIdent]
accumulators = [pairSum, pairMin, pairMax, pairAvg]

#featurenames = ["Cos", "Euc", "Mkow", "MaxDiff", "MinDiff"]
accnames = ["Sum", "Min", "Max", "Avg"]

allfeatures = {
		"Cos": wvCos,
		"City": wvCity,
		"Cheb": wvCheb,
		"Canb": wvCanb,
		"Corr": wvCorr,
		"Bray": wvBray,
		"Euc": wvEuc,
		"Mkow": wvMkow,
		"MinDiff": wvMinDiff,
		"MaxDiff": wvMaxDiff,
		"Ident": wvIdent,
		"Sum": wvSum,
		"Dot": wvDot,
		"Diff": wvDiff,
		"Mult": wvMult,
		"Std": wvStd,
}

#wvfeatures = set(["Ident", "Diff", "Mult"])
#singlefeatures = set(["Cos", "Euc", "Mkow", "MaxDiff", "MinDiff"])


def featurize(p, featurefns):
	a = []
	for accumulator in accumulators:
		a.append(accumulator(p))

	x = []
	for featurefn in featurefns:
		try:
			fs = featurefn(a)
			for f in fs:
				try:
					len(f)
					printd("Bad feature from %s" % featurefn.__name__, -1)
					printd(f, -1)
					sys.exit(-1)
				except TypeError:
					pass
			x.extend(fs)
		except Exception:
			print >>sys.stderr, "Pair:", p.pid
			print >>sys.stderr, "%d" % len(p.s1["vector"]), " ".join(p.s1["wv_tokens"])
			print >>sys.stderr, "%d" % len(p.s2["vector"]), " ".join(p.s2["wv_tokens"])
			traceback.print_exc()
			sys.exit(-1)

	if len(x) == 0 and len(featurefns) != 0:
		print >>sys.stderr, "Pair:", p.pid
		print >>sys.stderr, "%d" % len(p.s1["vector"]), " ".join(p.s1["wv_tokens"])
		print >>sys.stderr, "%d" % len(p.s2["vector"]), " ".join(p.s2["wv_tokens"])
		sys.exit(-1)
	return x


def sumproduct(a, b):
	for x, y in itertools.product(a, b):
		yield "%s_%s" % (x, y)


class FeatureSet(object):
	def __init__(self, data=None, featurenames=[]):
		self.train = []
		self.test = []
		self.trainlabels = []
		self.testlabels = []

		featurenames = sorted(featurenames)
		features = [allfeatures[x] for x in featurenames]

		if data is not None:
			self.data = data

			testa = [np.array([[0] * conf.wvsize, [0] * conf.wvsize])]

			names = ["Label"]
			for fname in featurenames:
				fnames = sumproduct([fname], range(len(allfeatures[fname](testa))))
				#Hack for badness...
				if fname == "Ident":
					names.extend(sumproduct(["Sum"], fnames))
				else:
					names.extend(sumproduct(accnames, fnames))
			self.names = names
			numfeat = len(names) - 1
			badfeat = [999] * numfeat
			#badfeat = np.array(numfeat)
			#badfeat.fill(999)

			for pair in data.train():
				self.trainlabels.append(pair.label)
				try:
					fs = featurize(pair, features)
				except ValueError:
					fs = list(badfeat)
				self.train.append(fs)
			for pair in data.test():
				self.testlabels.append(pair.label)
				try:
					fs = featurize(pair, features)
				except ValueError:
					fs = list(badfeat)
				self.test.append(fs)

		#self.trainlabels = np.array(self.trainlabels, dtype=np.float32)
		#self.testlabels = np.array(self.testlabels, dtype=np.float32)

	def discretizeLabels(self, minv=0, maxv=5, byweight=True):
		if minv is None or maxv is None:
			minv = floor(min(chain(self.trainlabels, self.testlabels)))
			maxv = ceil(max(chain(self.trainlabels, self.testlabels)))

		rng = maxv - minv + 1

		self.trainweights = [1] * len(self.train)
		#self.testweights = [1] * len(self.test)
		#for data, labels, weights in ((self.train, self.trainlabels, self.trainweights), (self.test, self.testlabels, self.testweights)):
		data = self.train
		labels = self.trainlabels
		weights = self.trainweights
		for i in range(len(data)):
			classes = [0] * rng
			v = labels[i]
			fl = int(floor(v))
			cl = int(ceil(v))
			if fl == cl:
				classes[fl] = 1
				weights[i] = 1
			else:
				newclasses = list(classes)
				# If close to the floor, should be higher weight, i.e.
				# 1 - (v - fl) = fl + 1 - v = cl - v
				flw = cl - v
				clw = v - fl
				if byweight:
					data.append(data[i])
					classes[fl] = 1
					newclasses[cl] = 1
					weights[i] = flw
					weights.append(clw)
					labels.append(newclasses)
				else:
					classes[fl] = flw
					classes[cl] = clw
			labels[i] = classes
		self.trainweights = np.asarray(self.trainweights, dtype=np.float32)

	def addMatcher(self, matcher, namebase=""):
		names = [namebase + x for x in matcher.names()]
		self.names.extend(names)
		for datum, pair in chain(izip(self.train, self.data.train()), izip(self.test, self.data.test())):
			try:
				matcher.match(pair)
			except ValueError, err:
				printd(err)
			fs = matcher.features()
			if len(names) != len(fs):
				printd("Incorrect names for features for %s: %d vs %d" % (matcher.__class__.__name__, len(names), len(fs)), -1)
				printd(names, -1)
				sys.exit(-1)

			for f in fs:
				# Checking
				if np.isnan(f) or np.isinf(f):
					printd("Bad feature from %s" % matcher.__class__.__name__, -1)
					printd(f, -1)
					printd(pair, -1)
					sys.exit(-1)
			datum.extend(fs)

	def write(self, outf):
		with open(outf + ".train", 'w') as fh:
			print >>fh, "\t".join(self.names)
			for label, x in izip(self.trainlabels, self.train):
				print >>fh, "%g\t" % label,
				print >>fh, "\t".join(["%g" % v for v in x])

		with open(outf + ".test", 'w') as fh:
			print >>fh, "\t".join(self.names)
			for label, x in izip(self.testlabels, self.test):
				print >>fh, "%g\t" % label,
				print >>fh, "\t".join(["%g" % v for v in x])

	@classmethod
	def read(cls, outf):
		fs = cls()

		with open(outf + ".train") as fh:
			fs.names = fh.readline().strip().split()
			# Check that it is actually a header, i.e. that the first element is not a number
			try:
				float(fs.names[0])
				fh.seek(0)
				fs.names = []
			except ValueError:
				pass
			for line in fh:
				p = line.strip().split()
				label = float(p[0])
				feats = [float(x) for x in p[1:]]
				fs.trainlabels.append(label)
				fs.train.append(feats)

		with open(outf + ".test") as fh:
			testnames = fh.readline().strip().split()
			try:
				float(testnames[0])
				fh.seek(0)
			except ValueError:
				pass
			for line in fh:
				p = line.strip().split()
				label = float(p[0])
				feats = [float(x) for x in p[1:]]
				fs.testlabels.append(label)
				fs.test.append(feats)

		return fs

	def freeze(self):
		self.train = np.asarray(self.train, dtype=np.float32)
		self.test = np.asarray(self.test, dtype=np.float32)
		self.trainlabels = np.asarray(self.trainlabels, dtype=np.float32)
		self.testlabels = np.asarray(self.testlabels, dtype=np.float32)


def linearize(labels):
	res = np.zeros((len(labels)))
	for i in range(len(labels)):
		# sum of the product of the index and the proportional weight for that index
		# divide by total weight to normalize?
		# Could also take the largest 2 and find their proportional weight
		#print >>sys.stderr, "\t".join(["%g" % x for x in labels[i]])
		#print >>sys.stderr, sum([x * y for x, y in enumerate(labels[i])]) / sum(labels[i])
		res[i] = sum([x * y for x, y in enumerate(labels[i])]) / sum(labels[i])
	return res


def evalData(labels, model=None, data=None, obs=None, classify=False, **predictargs):
	if obs is None:
		if classify:
			obs = model.predict_proba(data, **predictargs)
			obs = linearize(obs)
		else:
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

	return obs


def evalModel(model, data, labels):
	if conf.args.classify:
		obs = np.array(model.predict_proba(data)).T
		obs = linearize(obs)
	else:
		obs = model.predict(data)
	exp = labels

	pearsonsR, pearsonsP = pearsonr(obs, exp)
	return pearsonsR


def processData(args):
	data = dataset.Dataset.load(args.input_data, args.dataset)
	wvout = args.wvfile
	if os.path.exists(wvout):
		wordvecf = wvout
	else:
		wordvecf = args.wvsource

	features = {x for x in args.basefeatures.split(',') if x != ''}
	matchers = {x for x in args.matchers.split(',') if x != ''}

	printd("Loading Word Vectors")
	wordvec = WordVec(wordvecf)
	printd("Vectorizing")
	data.vectorize(wordvec)
	maxwords = data.maxShortSentence()

	if wvout != wordvecf:
		printd("Rereading word vectors to optimize...")
		wv_toks = data.wv_sentences()
		wordvec = WordVec(wordvecf, sentences=wv_toks, wvout=wvout, size=wordvec.originalsize)
		data.vectorize(wordvec)

	conf.wvsize = wordvec.size

	# Train data
	printd("Computing basic WV Features")
	fs = FeatureSet(data, features)

	if "Pair" in matchers:
		printd("Computing Pair Features")
		matcher = vectorsim.PairFeatures(dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher)

	if "Shingle" in matchers:
		printd("Computing Shingle Features")
		matcher = Shingler(slop=12, lmbda=0.95)
		fs.addMatcher(matcher)

	vocab = None
	if "MinDistSim" in matchers:
		printd("Computing MinDist")
		vocab = fs.data.wv_vocab()
		data.weight()
		comparator = 'cosine'
		matcher = vectorsim.MinDistSim(metric=comparator, df=vocab, maxsent=maxwords, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher, 'cos')
		printd("Computing MinDist-Euclidean")
		comparator = 'euclidean'
		matcher = vectorsim.MinDistSim(metric=comparator, df=vocab, maxsent=maxwords, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher, 'euc')

	if "NGram" in matchers:
		printd("Computing MinDist-Ngram")
		vocab = fs.data.wv_vocab()
		if vocab is None:
			vocab = fs.data.wv_vocab()
		comparator = 'cosine'
		matcher = vectorsim.MinDistSim(metric=comparator, df=vocab, maxsent=maxwords, ngram=2, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher, 'cos-bigram')
		comparator = 'cosine'
		matcher = vectorsim.MinDistSim(metric=comparator, df=vocab, maxsent=maxwords, ngram=3, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher, 'cos-trigram')

	if "WWSim" in matchers:
		printd("Computing WWSim")
		matcher = vectorsim.WWSim(wordvec=wordvec, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher)

	if "InfRankSim" in matchers:
		printd("Computing InfRankSim")
		matcher = vectorsim.InfRankSim(data=data, wordvec=wordvec, dimfeatures=args.dimfeatures)
		printd("InfRankSim Matching")
		fs.addMatcher(matcher)

	if "InfSim" in matchers:
		# We normalize after so primary features are raw word vectors
		# InfSim
		printd("Computing InfSim")
		wordvec.normalize()
		data.vectorize(wordvec)
		matcher = vectorsim.InfSim(data=data, wordvec=wordvec, dimfeatures=args.dimfeatures)
		fs.addMatcher(matcher)

	return fs


param_grids = {
		"randomforests":
		{'max_depth': [1, 3, 10, 300],
			'n_estimators': [30, 100, 300, 500],
			'max_features': ['sqrt', 0.5, 0.8, 1.0],
			'min_samples_split': [2, 4, 8],
			},
		"extratrees":
		{'max_depth': [1, 3, 10, 300],
			'n_estimators': [100, 300, 500, 1000, 5000],
			'max_features': ['sqrt', 0.5, 0.8, 1.0],
			'min_samples_split': [2, 4, 8],
			},
		"gradientboosting":
		{'max_depth': [1, 3, 10, 300],
			'n_estimators': [30, 100, 300, 500],
			'max_features': ['sqrt', 0.5, 0.8, 1.0],
			'min_samples_split': [2, 4, 8],
			'subsample': [0.5, 0.8, 1.0],
			},
}

default_params = {
		"randomforests":
		{
			'max_depth': 20,
			#'max_depth': None,
			#'max_leaf_nodes': 20,
			'max_leaf_nodes': None,
			'n_estimators': 800,
			'max_features': 'sqrt',
			'min_samples_split': 4,
			'oob_score': True,
			'n_jobs': -1,
			#'criterion': 'mae',
			},
		"extratrees":
		{
			'max_depth': 300,
			'n_estimators': 300,
			'max_features': 0.5,
			'min_samples_split': 4,
			'oob_score': True,
			'bootstrap': True,
			'n_jobs': -1,
			},
		"gradientboosting":
		{
			'max_depth': 3,
			'n_estimators': 500,
			'max_features': 0.5,
			'min_samples_split': 4,
			'subsample': 1.0,
			},
		"adaboost":
		{
			'n_estimators': 500,
			'learning_rate': 1.0,
			'loss': 'linear',
			},
		"linreg":
		{
			'fit_intercept': True,
			'normalize': True,
			},
		"decisiontree":
		{
			'max_depth': 3,
			'max_features': 0.5,
			'min_samples_split': 4,
			},
		"autolearn":
		{
			},
		"nn":
		{
			"input_dim": None
			},
}


def NNModel(input_dim, dropout=1, n_classes=6, n_hidden=64):
	from keras.models import Sequential
	from keras.layers import Dense, Activation, Dropout
	model = Sequential()
	if dropout < 1 and dropout > 0:
		model.add(Dropout(p=dropout))
	model.add(Dense(output_dim=n_hidden, input_dim=input_dim))
	model.add(Activation("sigmoid"))
	model.add(Dense(output_dim=n_classes, input_dim=n_hidden))
	model.add(Activation("softmax"))
	model.compile(loss='kld', optimizer='sgd', metrics=['accuracy'])
	return model


def main(args):
	conf.debug = args.debug
	conf.verbose = args.verbose
	conf.args = args
	if args.alphas == "auto":
		pass  # TODO
	else:
		conf.alphas = [float(x) for x in args.alphas.split(",")]

	try:
		args.max_features = int(args.max_features)
	except (ValueError, TypeError):
		try:
			args.max_features = float(args.max_features)
		except (ValueError, TypeError):
			pass

	fitargs = {}
	predictargs = {}

	if args.featurefile and not args.force and os.path.exists(args.featurefile + ".train"):
		printd("Loading Saved Features")
		fs = FeatureSet.read(args.featurefile)
	else:
		fs = processData(args)
		if args.featurefile:
			printd("Writing Features")
			fs.write(args.featurefile)

	#kpca = KernelPCA(kernel="rbf", gamma=10)
	if args.model == "randomforests":
		if args.classify:
			model = ensemble.RandomForestClassifier
		else:
			model = ensemble.RandomForestRegressor
	elif args.model == "extratrees":
		if args.classify:
			model = ensemble.ExtraTreesClassifier
		else:
			model = ensemble.ExtraTreesRegressor
	elif args.model == "gradientboosting":
		if args.classify:
			model = ensemble.GradientBoostingClassifier
		else:
			model = ensemble.GradientBoostingRegressor
	elif args.model == "decisiontree":
		model = DecisionTreeRegressor
	elif args.model == "adaboost":
		model = ensemble.AdaBoostRegressor
	elif args.model == "linreg":
		model = LinearRegression
	elif args.model == "autolearn":
		printd("AutoLearn disabled as it does not work properly")
		sys.exit(-1)
		#model = AutoSklearnClassifier
		fitargs["dataset_name"] = "semeval"
	elif args.model == "nn":
		model = NNModel
		fitargs["nb_epoch"] = 10
		fitargs["batch_size"] = 32
		fitargs["verbose"] = 2
		predictargs["verbose"] = 0
	elif args.model == "None":
		printd("No Model specified, exiting")
		sys.exit(-1)
	else:
		printd("Invalid model %s" % args.model)
		sys.exit(-1)

	if args.classify:
		# Forest Classifiers do not allow non-binary labels, so we do it by sample weight instead
		byweight = issubclass(model, ensemble.forest.ForestClassifier)
		lintrainlabels = np.copy(fs.trainlabels)
		fs.discretizeLabels(byweight=byweight)
		if byweight:
			fitargs["sample_weight"] = fs.trainweights
	else:
		lintrainlabels = np.array(fs.trainlabels)
	fs.freeze()
	printd("Train labels:" + str(fs.trainlabels.shape))

	if (not args.force) and args.modelfile and os.path.exists(args.modelfile):
		if args.model == "nn":
			import keras
			model = keras.models.load_model(args.modelfile)
		else:
			model = joblib.load(args.modelfile)
	else:
		params = default_params[args.model]
		for param_name, param_value in params.items():
			try:
				pval = getattr(args, param_name)
				if pval is not None:
					params[param_name] = pval
			except AttributeError:
				pass

		if "input_dim" in params:
			# -1 for label
			params["input_dim"] = len(fs.names) - 1
		model = model(**params)

		if args.gridsearch:
			model = GridSearchCV(model, scoring=evalModel, cv=5, error_score=0,
					param_grid=param_grids[args.model], n_jobs=16, pre_dispatch="2*n_jobs", verbose=10)
		#model = Pipeline(steps=[('pca', kpca), ('dtree', dtree)])
		printd("Training")
		model.fit(fs.train, fs.trainlabels, **fitargs)
		#X_kpca = kpca.fit_transform(X)
		#dtree.fit(traindata, trainlabels)
		if args.modelfile:
			try:
				if args.model == "nn":
					model.save(args.modelfile)
				else:
					joblib.dump(model, args.modelfile)
			except Exception:
				printd("Could not save model, autolearn does not support saving")

	printd("Evaluating")
	print "Using Features: %s" % args.basefeatures
	print "Using Matchers: %s" % args.matchers
	print "Train Accuracy"
	evalData(model=model, data=fs.train, labels=lintrainlabels, classify=args.classify, obs=model.oob_prediction_, **predictargs)
	# trainobs = _

	print "Test Accuracy"
	testobs = evalData(model=model, data=fs.test, labels=fs.testlabels, classify=args.classify, **predictargs)

	if args.writematches:
		try:
			fs.data.writer
		except AttributeError:
			fs.data = dataset.Dataset.load(args.input_data, args.dataset)
		trainwriter = fs.data.writer(args.writematches + ".train")
		testwriter = fs.data.writer(args.writematches + ".test")
		for pair in fs.data.train():
			trainwriter.write(pair, Match(score=pair.label, autop=0))
		for pair, obs in izip(fs.data.test(), testobs):
			testwriter.write(pair, Match(score=obs))

	#print "Combined Accuracy"
	#alllabels = np.concatenate((lintrainlabels, fs.testlabels))
	#allobs = np.concatenate((trainobs, testobs))
	#evalData(labels=alllabels, obs=allobs, classify=args.classify, **predictargs)


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
		"alphas": "auto",
		"comparator": "cosine",
		"dataset": "auto",
		"input_data": "../results/nuggets.tsv",
		"limit": None,
		#"max_features": "auto",
		#"basefeatures": "Cos,Euc,Mkow,MaxDiff,MinDiff,Mult,Ident,Diff",
		#"basefeatures": "Cos,Euc,Mkow,MaxDiff,MinDiff,Mult,Sum,Diff",
		"basefeatures": "Cos,City,Cheb,Canb,Corr,Bray,Euc,Mkow,MinDiff,MaxDiff,Sum,Dot,Diff,Mult,Std",
		#"matchers": "InfSim,InfRankSim,MinDistSim,NGram,WWSim,Pair",
		"matchers": "InfSim,InfRankSim,MinDistSim,NGram,WWSim,Pair,Shingle",
		#"min_samples_split": 2,
		"model": "randomforests",
		"neg_samples": 5,
		#"n_estimators": 40,
		"sim": "sum",
		"dimfeatures": True,
		"sim_thr": -10,
		"wvsource": "/huge1/sourcedata/word2vec/google/GoogleNews-vectors-negative300.bin",
		}
	if args.conf_file:
		config = ConfigParser.SafeConfigParser()
		config.read([args.conf_file])
		defaults.update(dict(config.items("Defaults")))

	argparser = argparse.ArgumentParser(description='Automatically match using shingles and word vectors', parents=[conf_parser])
	argparser.set_defaults(**defaults)
	argparser.add_argument('-i', '--input_data', help='Dataset file/dir')
	argparser.add_argument('--dataset', help='The name of the dataset (otherwise inferred from files)')
	argparser.add_argument('-d', '--debug', action='store_true', help='Debug mode (lots of output)')
	argparser.add_argument('--verbose', action='store_true', help='Even more output')
	argparser.add_argument('-v', '--wordvec', help='Write out wordvec to file/dir')
	argparser.add_argument('--model', help='Type of model to use:' +
			'randomforests, gradientboosting, decisiontree, extratrees, autolearn, adaboost, or None (do not learn)')
	argparser.add_argument('--modelfile', help='Output location for model, reused if exists unless -f is used.')
	argparser.add_argument('--featurefile', help='Write out features to file, if does exists')
	argparser.add_argument('--matchers', help='Matching Features to use (comma separated list, no spaces).')
	argparser.add_argument('--basefeatures', help='Base features to use (comma separated list, no spaces).')
	argparser.add_argument('--frequencies', help='Read from word frequencies JSON file')
	argparser.add_argument('-f', '--force', action='store_true', help='Relearning and rewriting files')
	argparser.add_argument('--gridsearch', action='store_true', help='Run Grid Search for learning function hyperparameters')
	argparser.add_argument('--classify', action='store_true', help='Perform classification instead of regression.')
	argparser.add_argument('--max_depth', type=int, help='Max depth of a tree')
	argparser.add_argument('--max_leaf_nodes', type=int, help='Max number of leaf nodes')
	argparser.add_argument('--neg_samples', type=int, help='Number of negative samples to use for all-pairs datasets (default 5)')
	argparser.add_argument('--alphas', help='Weight of each wordvec')
	argparser.add_argument('--min_samples_split', type=int, help='Min Samples to split on (for trees)')
	argparser.add_argument('--n_estimators', type=int, help='Max number of estimators (for ensembles)')
	argparser.add_argument('--max_features', help='Max number of features to consider for split (for ensembles)')
	argparser.add_argument('--dimfeatures', action='store_true', help='Use dimensional features in the matchers.')
	argparser.add_argument('--no-dimfeatures', dest="dimfeatures", action='store_false', help='Do not use dimensional features in the matchers.')
	argparser.add_argument('-w', '--wvfile', help='Store minimal version of wordvec file after processing, ' +
			'for faster subsequent runs (reused if exists, unless -f used).')
	argparser.add_argument('--wvsource', help='Source file for wordvecs.')
	argparser.add_argument('--learn', help='Write features for a learning algorithm')
	argparser.add_argument('--logfile', help='Log to file (in addition to stderr).')
	argparser.add_argument('--writematches', help='Write matches to given file.')
	main(argparser.parse_args(remaining_argv))
