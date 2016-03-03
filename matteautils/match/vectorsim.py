#!/usr/bin/python
import numpy as np
from scipy.spatial.distance import cdist
from munkres import munkres
from . import Matcher

#a = np.array([[.1]*5,[.2]*5,[.3]*5])
#b = np.array([[.1]*5,[.2,.2,.2,.2,.3],[0,0,0,0,1]])
#c = 1 - cdist(a,b,'cosine')


class MinDist(Matcher):

	def __init__(self, s1, s2):
		self.dist = dist(s1, s2)
	
	def match(self):
		matches = munkres(self.dist)

		tdist = 0
		for i in range(matches.shape[0]):
			for j in range(matches.shape[1]):
				if matches[i,j]:
					tdist += self.dist[i,j]
		return tdist / max(matches.shape)


# Lots of possible fns:
# http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.spatial.distance.cdist.html
def dist(s1, s2, fn='cosine'):
	return cdist(s1, s2, fn)
