import sys
import csv
import fileinput
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.probability import *
import types

pos_stmt = []
neg_stmt = []

"""
pos_reader = csv.reader(open('Yes.csv','rb'))
for row in pos_reader:
	pos_stmt.append((row[0],'yes'))		
#print pos_stmt

neg_reader = csv.reader(open('No.csv','rb'))
for row in neg_reader:
	neg_stmt.append((row[0],'no'))
#print  neg_stmt
"""
try:
	#Parse CSV
	csvparse = csv.reader(open('DataSet.csv','rb'))
	#Go Row by Row, identify Positive and Negative stmts
	for row in csvparse:
		if row[1] == 'Yes':
			pos_stmt.append((row[0],row[1]))
		else:
			neg_stmt.append((row[0],row[1]))

	#print pos_stmt
	#print neg_stmt

	stmt = []
	for (words, sentiment) in pos_stmt + neg_stmt:
	    words_filtered = [e.lower() for e in words.split() if len(e) >= 0]
	    stmt.append((words_filtered, sentiment))

	#get all words
	def get_words_in_tweets(stmt):
	    all_words = []
	    for (words, sentiment) in stmt:
	        all_words.extend(words)
	    return all_words

	#get word in order of frequency, to track which one comes how many times and its labeled with what (wordlist)
	def get_word_features(wordlist):
	    wordlist = nltk.FreqDist(wordlist)
	    word_features = wordlist.keys()
	    return word_features

	#first get list of words in tweet, based on frequency distribution get word_features
	word_features = get_word_features(get_words_in_tweets(stmt))
	
	#print word_features
	def extract_features(document):
	    document_words = set(document)
	    features = {}
	    for word in word_features:
	        features['contains(%s)' % word] = (word in document_words)
	    return features
	
	training_set = nltk.classify.util.apply_features(extract_features, stmt)
	#print training_set	
	classifier = nltk.NaiveBayesClassifier.train(training_set)
		
	#Print individual's probability
#	print label_probdist.prob('Yes')
#	print label_probdist.prob('No')
	
	#Check the accuracy of the classifier
	print nltk.classify.accuracy(classifier, training_set)
	
	f_set =  classifier.labels()
	print f_set
	#print classifier.prob_classify({"No":True})
	#Print top 50 most informative words based on which its straight to figure out underlying positive/negative feelings
	print classifier.show_most_informative_features(250)
	
	for line in fileinput.input("sample.txt"):
		print line + ':' + classifier.classify(extract_features(line.split()))
		
except:
	print "Unexpected error:", sys.exc_info()[0]
	raise

