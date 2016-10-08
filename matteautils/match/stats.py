import operator as op
import scipy.special as special
import numpy as np


def tauap(ideal, exp, reverse=False):
	if (len(ideal) != len(exp)):
		raise Exception("lists of differing size")
	if reverse:
		cmpfn = op.gt
	else:
		cmpfn = op.lt
	n = len(ideal)
	lst = sorted(zip(ideal, exp), key=op.itemgetter(0), reverse=reverse)
	tap = 0
	for ind in range(1, len(lst)):
		pos = 0
		for subind in range(0, ind):
			if cmpfn(lst[subind][1], lst[ind][1]):
				pos += 1
		tap += pos / ind
	tap = 2 / (n - 1) * tap - 1
	svar = (4.0 * n + 10.0) / (9.0 * n * (n - 1))
	z = tap / np.sqrt(svar)
	prob = special.erfc(np.abs(z) / 1.4142136)
	return tap, prob
