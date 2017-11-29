from pprint import pprint
from time import time
import logging
import numpy as np 
import sys
import math
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
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

def run_lda(preprocess, data):
	lda = preprocess.get_params()['lda__lda_model']
	preprocess.fit(data)
	gamma, perplexity = preprocess.transform(data)
	return perplexity

def validate(preprocess, C, training_data, training_target, val_data, val_target):
	# Preprocess
	preprocess.fit(training_data)
	training_features = preprocess.transform(training_data)
	val_features = preprocess.transform(val_data)
	# Validate
	train_scores = []
	val_scores = []
	for c in C:
		clf = LinearSVC(max_iter=500, C=c)
		clf.fit(training_features, training_target)
		train_predict = clf.predict(training_features)
		train_score = f1_score(training_target, train_predict, average='macro')
		val_predict = clf.predict(val_features)
		val_score = f1_score(val_target, val_predict, average='macro')
		train_scores.append(train_score)
		val_scores.append(val_score)
	return train_scores, val_scores

def learning(pipeline, train_data, train_target, test_data, test_target):
	pipeline.fit(train_data, train_target)
	test_pred = pipeline.predict(test_data)
	f1 = f1_score(test_target, test_pred, average='macro')
	return f1

def multi_run_wrapper(tup):
   return tup[0](*tup[1])

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

	############################# Vectorizer
	vect_result = 'max_df=.75, ngram_range=(1, 1), max_features=30000'

	############################# Tune LDA
	V = 30000
	kappa = 0.5
	tau0 = 64
	var_i = 100
	num_topics = 20

	# pool = Pool(processes=3)
	# works = []
	# params = []
	# for size in [256, 512]:
	# 	for alpha in [.1, .3, .5, .7]:

	# 		preprocess = Pipeline([
	# 			('count', CountVectorizer(stop_words='english', 
	# 							max_df=.75, ngram_range=(1, 1), max_features=V)),
	# 			('lda', LDAVectorizer(num_topics=num_topics, V=V, alpha=alpha,
	# 							kappa=kappa, tau0=tau0, var_i=var_i, size=size))
	# 		])

	# 		works.append((run_lda, (preprocess, train_data)))
	# 		params.append({ 'num_topics': num_topics, 'size': size, 'alpha': alpha})

	# perplexities = pool.map(multi_run_wrapper, works)
	# pool.close()
	# pool.join()
	# make_dir('result/svm-lda/%d/' % num_topics)
	# with open('result/svm-lda/%d/lda-tune' % num_topics, 'w') as f:
	# 	f.write(str(params))
	# 	f.write(str(perplexities))
	# save_pickle((params, perplexities), 'result/svm-lda/%d/lda-tune.pkl' % num_topics)	

	############################ Best preprocessor
	# params, perplexities = load_pickle('result/svm-lda/%d/lda-tune.pkl' % num_topics)
	# imin = np.argmin(perplexities)
	# best_lda_params = params[imin]

	best_lda_params = { 'num_topics': num_topics, 'size': 256, 'alpha': .1} # 20
	# best_lda_params = { 'num_topics': num_topics, 'size': 512, 'alpha': .1} # 50

	############################ Cross validation
	skf = StratifiedKFold(n_splits=3)
	# C = np.logspace(-3, 0, 10)
	C = np.linspace(0.01, 0.1, 10)
	val_dir = 'C = np.linspace(0.01, 0.1, 10)'
	works = []
	for train_ids, val_ids in skf.split(train_data, train_target):
		# Tuned preprocessor
		best_preprocessor = Pipeline([
			('count', CountVectorizer(stop_words='english', 
							max_df=.75, ngram_range=(1, 1), max_features=V)),
			('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
						alpha=best_lda_params['alpha'],
						kappa=kappa, tau0=tau0, var_i=var_i, 
						size=best_lda_params['size'], perplexity=False))
		])

		# Preprocess
		training_data = [train_data[idx] for idx in train_ids]
		training_target = [train_target[idx] for idx in train_ids]
		val_data = [train_data[idx] for idx in val_ids]
		val_target = [train_target[idx] for idx in val_ids]

		# Work
		works.append((validate, (best_preprocessor, C, training_data, training_target, val_data, val_target)))
		
	# Run processes
	pool = Pool(processes=3)
	cv_results = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()	
	# Result
	train_scores = [res[0] for res in cv_results]
	val_scores = [res[1] for res in cv_results]
	mean_train_score = np.mean(train_scores, axis=0)
	mean_val_score = np.mean(val_scores, axis=0)
	# Refit
	imax = np.argmax(mean_val_score)
	params = {'C': C[imax]}
	best_preprocessor = Pipeline([
		('count', CountVectorizer(stop_words='english', 
						max_df=.75, ngram_range=(1, 1), max_features=V)),
		('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
					alpha=best_lda_params['alpha'],
					kappa=kappa, tau0=tau0, var_i=var_i, 
					size=best_lda_params['size'], perplexity=False))
	])
	best_preprocessor.fit(train_data)
	save_pickle(best_preprocessor, 'result/svm-lda/%d/pre.pkl' % (num_topics))
	best_preprocessor = load_pickle('result/svm-lda/%d/pre.pkl' % (num_topics))
	train_features = best_preprocessor.transform(train_data)
	best_clf = LinearSVC(max_iter=500)
	best_clf.set_params(**params)
	best_clf.fit(train_features, train_target)
	save_pickle(best_clf, 'result/svm-lda/%d/clf.pkl' % (num_topics))
	best_clf = load_pickle('result/svm-lda/%d/clf.pkl' % (num_topics))
	make_dir('result/svm-lda/%d/%s/' % (num_topics, val_dir))
	with open('result/svm-lda/%d/%s/cvresult.txt' % (num_topics, val_dir), 'w') as f:
		f.write('Params:\n')
		f.write(str({'C': C}))
		f.write('\nmean_train_score:\n')
		f.write(str(mean_train_score)) 
		f.write('\nmean_val_score:\n')
		f.write(str(mean_val_score)) 
	# Plot
	plt.xlabel('C')
	plt.ylabel('Score')
	plt.plot(C, mean_train_score, c='r', label='Training score')
	plt.plot(C, mean_val_score, c='b', label='Validation score')
	plt.legend()
	# plt.show()
	plt.savefig('result/svm-lda/%d/%s/val_curve.png' % (num_topics, val_dir))
	plt.close()
	############################ Test
	print('----------- Test')
	test = fetch_20newsgroups(subset='test')
	test_data = load_pickle('dataset/test-data.pkl')[:]
	test_target = test.target[:]
	test_features = best_preprocessor.transform(test_data)
	test_pred = best_clf.predict(test_features)
	test_score = f1_score(test_target, test_pred, average='macro')
	with open('result/svm-lda/%d/report' % num_topics, 'w') as f:
		f.write('Best preprocessor:\n')
		f.write(str(best_preprocessor.get_params()))
		f.write('\n\n\n')
		f.write('Best classifier:\n')
		f.write(str(best_clf.get_params()))
		f.write('\n\n\n')
		f.write(classification_report(test_target, test_pred))

	############################ Learning curve
	n_train = len(train_target)
	percent = np.linspace(0.1, 1, 10)
	pool = Pool(processes=3)
	works = []
	for r in percent:
		best_preprocessor = Pipeline([
			('count', CountVectorizer(stop_words='english', 
							max_df=.75, ngram_range=(1, 1), max_features=V)),
			('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
						alpha=best_lda_params['alpha'],
						kappa=kappa, tau0=tau0, var_i=var_i, 
						size=best_lda_params['size'], perplexity=False))
		])
		best_clf = load_pickle('result/svm-lda/%d/clf.pkl' % (num_topics))
		pipeline = Pipeline([
			('pre', best_preprocessor),
			('clf', best_clf)
		])

		n = int(r * n_train)
		works.append((learning, (pipeline, train_data[:n], train_target[:n], \
					test_data, test_target)))

	f1 = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()
	save_pickle((percent, f1), 'result/svm-lda/%d/learning' % num_topics)
	# Plot
	plt.xlabel('Percent of train data')
	plt.ylabel('F1 score')
	plt.plot(percent, f1, c='r', label='Test score')
	plt.legend()
	plt.savefig('result/svm-lda/%d/learning_curve.png' % num_topics)
	# plt.show()
	

