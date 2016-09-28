import csv
import sys
import os
import codecs
import glob
import config as conf

__all__ = ["TSVReader", "UnicodeDictReader", "printd", "Tee"]


def UnicodeDictReader(utf8_data, **kwargs):
	csv_reader = csv.DictReader(utf8_data, **kwargs)
	for row in csv_reader:
		yield {key: unicode((value or ""), 'utf-8') for key, value in row.iteritems()}


#def utf_8_encoder(unicode_csv_data):
#	for line in unicode_csv_data:
#		yield line.encode('utf-8')


def bom_stripper(fh):
	pos = fh.tell()
	for line in fh:
		if pos == 0:
			line = line.replace(codecs.BOM_UTF8, "")
		yield line
		pos = fh.tell()


def TSVReader(filen, **kwargs):
	#with codecs.open(filen, 'r', encoding='utf-8-sig') as fh:
	with open(filen, 'r') as fh:
		csv_reader = csv.DictReader(bom_stripper(fh), delimiter="\t", quoting=csv.QUOTE_NONE, **kwargs)
		for row in csv_reader:
			yield {key: unicode(value, 'utf-8') for key, value in row.iteritems()}


def getfiles(filenames):
	if type(filenames) == str:
		if os.path.exists(filenames):
			filenames = [filenames]
		else:
			filenames = glob.glob(filenames)
	return filenames


def printd(string, level=0):
	if level < 0 or (level == 0 and conf.debug) or conf.verbose:
		print >> sys.stderr, string


def traverse(o, tree_types=(list, tuple)):
	if isinstance(o, tree_types):
		for value in o:
			for subvalue in traverse(value, tree_types):
				yield subvalue
	else:
		yield o


class Tee(object):
	def __init__(self, name, mode='w', pipe='stdout', **kwargs):
		if isinstance(name, basestring):
			self.fileh = codecs.open(name, mode=mode, encoding='utf-8', errors='ignore', **kwargs)
		else:
			self.fileh = name
		self.pipe = pipe
		self.pipeh = getattr(sys, pipe)
		setattr(sys, pipe, self)

	def __del__(self):
		setattr(sys, self.pipe, self.pipeh)
		self.fileh.close()

	def write(self, data):
		self.fileh.write(data)
		self.pipeh.write(data)
