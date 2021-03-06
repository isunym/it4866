from pprint import pprint
from time import time
import logging
import numpy as np 
import sys
import math
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from util import load_pickle, save_pickle, make_dir
from sklearn.model_selection import learning_curve, validation_curve,\
			cross_val_score, GridSearchCV, StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import json
from lib.document import Document 
from lib.lda_vectorizer import LDAVectorizer, count_matrix_to_documents
from multiprocessing import Pool

############################# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def multi_run_wrapper(tup):
   return tup[0](*tup[1])

def learning(lda, train_data, train_target, test_data, test_target, num_classes=20):
	D_train = len(train_data)
	lda.fit(train_data)
	train_features = lda.transform(train_data)
	num_topics = len(train_features[0])
	class_topic = np.zeros((num_classes, num_topics))
	for i in range(D_train):
		class_topic[train_target[i], :] += train_features[i]
	test_features = lda.transform(test_data) # DxK
	test_score_matrix = np.dot(test_features, class_topic.T) # DxC
	test_predict = np.argmax(test_score_matrix, axis=1)
	f1 = f1_score(test_target, test_predict, average='macro')
	return f1

if __name__ == '__main__':
	print('Loading 20newsgroup dataset for all categories')

	############################# Load train data
	train = fetch_20newsgroups(subset='train')
	print('Train data:\n')
	print('%d documents' % len(train.filenames))
	print('%d categories' % len(train.target_names))

	train_data = load_pickle('dataset/train-data.pkl')[:]
	train_target = train.target[:]
	D_train = len(train_target)

	############################# Count vectorizer
	vect_result = 'max_df=.75, ngram_range=(1, 1), max_features=30000'

	############################# Tuned LDA
	V = 30000
	kappa = 0.5
	tau0 = 64
	var_i = 100
	num_topics = 50

	# best_lda_params = { 'num_topics': num_topics, 'size': 256, 'alpha': .1}
	best_lda_params = { 'num_topics': num_topics, 'size': 512, 'alpha': .1}

	best_lda = Pipeline([
		('count', CountVectorizer(stop_words='english', 
						max_df=.75, ngram_range=(1, 1), max_features=V)),
		('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
					alpha=best_lda_params['alpha'],
					kappa=kappa, tau0=tau0, var_i=var_i, 
					size=best_lda_params['size'], perplexity=False))
	])

	make_dir('result/lda/%d' % num_topics)
	best_lda.fit(train_data)
	save_pickle(best_lda, 'result/lda/%d/lda.pkl' % (num_topics))
	best_lda = load_pickle('result/lda/%d/lda.pkl' % (num_topics))
	# best_lda = load_pickle('result/svm-lda/%d/pre.pkl' % (num_topics))
	train_features = best_lda.transform(train_data)

	############################# Top words
	# top_idxs = lda_model.get_top_words_indexes()
	# with open('result/lda/%d/top-words.txt' % num_topics, 'w') as f:
	# 	for i in range(len(top_idxs)):
	# 		s = '\nTopic %d:' % i 
	# 		for idx in top_idxs[i]:
	# 			s += ' %s' % inverse_vocab[idx]
	# 		f.write(s)

	############################# Class - topic matrix
	num_classes = len(train.target_names)
	class_topic = np.zeros((num_classes, num_topics))
	for i in range(D_train):
		class_topic[train_target[i], :] += train_features[i]

	class_topic = 1. * class_topic / np.sum(class_topic, axis=1).reshape(num_classes, 1) # CxK
	save_pickle(class_topic, 'result/lda/%d/class-topic.pkl' % num_topics)

	class_topic = load_pickle('result/lda/%d/class-topic.pkl' % num_topics)
	# Train predict
	train_score_matrix = np.dot(train_features, class_topic.T) # DxC
	train_predict = np.argmax(train_score_matrix, axis=1)

	print(classification_report(train_target, train_predict))

	############################# Load test data
	test = fetch_20newsgroups(subset='test')
	test_data = load_pickle('dataset/test-data.pkl')[:]
	test_target = test.target[:]
	D_test = len(test_target)

	test_features = best_lda.transform(test_data) # DxK

	test_score_matrix = np.dot(test_features, class_topic.T) # DxC
	test_predict = np.argmax(test_score_matrix, axis=1)
	test_score = f1_score(test_target, test_predict, average='macro')

	with open('result/lda/%d/report' % num_topics, 'w') as f:
		f.write('Best lda:\n')
		f.write(str(best_lda.get_params()))
		f.write('\n\n\n')
		f.write(classification_report(test_target, test_predict))

	############################ Learning curve
	n_train = len(train_target)
	percent = np.linspace(0.1, 1, 10)
	pool = Pool(processes=3)
	works = []
	for r in percent:
		best_lda = Pipeline([
			('count', CountVectorizer(stop_words='english', 
							max_df=.75, ngram_range=(1, 1), max_features=V)),
			('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
						alpha=best_lda_params['alpha'],
						kappa=kappa, tau0=tau0, var_i=var_i, 
						size=best_lda_params['size'], perplexity=False))
		])
		n = int(r * n_train)
		works.append((learning, (best_lda, train_data[:n], train_target[:n], \
					test_data, test_target, num_classes)))

	f1 = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()
	save_pickle((percent, f1), 'result/lda/%d/learning' % num_topics)
	# Plot
	plt.xlabel('Percent of train data')
	plt.ylabel('F1 score')
	plt.plot(percent, f1, c='r', label='Test score')
	plt.legend()
	plt.savefig('result/lda/%d/learning_curve.png' % num_topics)
	# plt.show()
		