import csv
import sys

__DEBUG__ = False

def UnicodeDictReader(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {key: unicode(value, 'utf-8') for key, value in row.iteritems()}

def printd(string):
    if __DEBUG__:
        print >> sys.stderr, string

def setdebug(debug=True):
	global __DEBUG__
	__DEBUG__ = debug
