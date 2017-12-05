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
			cross_val_score, GridSearchCV, StratifiedKFold, KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import json
from lib.document import Document 
from lib.lda_vectorizer import LDAVectorizer, count_matrix_to_documents
from multiprocessing import Pool
np.random.seed(0)

############################# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def tune_lda(training_data, val_data, V, alphas, sizes, params):
	count_vect = CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)
	count_vect.fit(training_data)
	training_features = count_vect.transform(training_data)
	val_features = count_vect.transform(val_data)
	val_perplexity = []
	val_params = []
	for alpha in alphas:
		for size in sizes:
			np.random.seed(0)
			lda_vect = LDAVectorizer(V=V, alpha=alpha, size=size, **params)
			lda_vect.fit(training_features)
			_, perplexity = lda_vect.transform(val_features)
			val_perplexity.append(perplexity)
			val_params.append({ 'alpha': alpha, 'size': size })

	return val_perplexity, val_params	

def validate(preprocess, C, training_data, training_target, val_data, val_target):
	# Preprocess
	preprocess.fit(training_data)
	training_features = preprocess.transform(training_data)
	val_features = preprocess.transform(val_data)
	# Validate
	train_scores = []
	val_scores = []
	for c in C:
		clf = LinearSVC(max_iter=500, C=c, loss='hinge')
		clf.fit(training_features, training_target)
		train_predict = clf.predict(training_features)
		train_score = f1_score(training_target, train_predict, average='weighted')
		val_predict = clf.predict(val_features)
		val_score = f1_score(val_target, val_predict, average='weighted')
		train_scores.append(train_score)
		val_scores.append(val_score)
	return train_scores, val_scores

def learning(pre, clf, train_data, train_target, test_data, test_target):
	
	pre.fit(train_data, train_target)
	train_features = pre.transform(train_data)
	clf.fit(train_features, train_target)
	test_features = pre.transform(test_data)
	test_pred = clf.predict(test_features)
	f1 = f1_score(test_target, test_pred, average='weighted')
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

	############################# Tune LDA
	V = 10000
	kappa = 0.5
	tau0 = 64
	var_i = 100
	num_topics = 20
	sizes = [512, 256]
	alphas = [.1, .05, .01]

	pool = Pool(processes=3)
	works = []
	kf = KFold(n_splits=3)
	lda_params = {
		'kappa': kappa,
		'tau0': tau0,
		'var_i': var_i,
		'num_topics': num_topics
	}
	for train_ids, val_ids in kf.split(train_data, train_target):
		training_data = [train_data[idx] for idx in train_ids]
		val_data = [train_data[idx] for idx in val_ids]

		works.append((tune_lda, (training_data, val_data, V, alphas, sizes, lda_params)))

	result = pool.map(multi_run_wrapper, works)	
	pool.close()
	pool.join()
	perplexities = [r[0] for r in result]
	avg_perplexities = np.mean(perplexities, axis=0)
	val_params = result[0][1]
	# LDA tune result
	make_dir('result/lda/%d/' % num_topics)
	with open('result/lda/%d/lda-tune' % num_topics, 'w') as f:
		f.write(str(val_params))
		f.write(str(avg_perplexities))
	save_pickle((val_params, avg_perplexities), 'result/lda/%d/lda-tune.pkl' % num_topics)	
	val_params, avg_perplexities = load_pickle('result/lda/%d/lda-tune.pkl' % num_topics)
	
	imin = np.argmin(avg_perplexities)
	best_lda_params = val_params[imin]

	############################ Cross validation
	np.random.seed(0)
	skf = StratifiedKFold(n_splits=3)
	# C = np.logspace(-3, 0, 10)
	C = np.linspace(0.01, 0.1, 10)
	# C = np.linspace(.2, .5, 10)
	val_dir = 'C = np.linspace(0.01, 0.1, 10)'
	works = []
	for train_ids, val_ids in skf.split(train_data, train_target):
		# Tuned preprocessor
		np.random.seed(0)
		best_preprocessor = Pipeline([
			('count', CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)),
			('lda', LDAVectorizer(num_topics=num_topics, V=V, 
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
	plt.savefig('result/svm-lda/%d/%s/val_curve.png' % (num_topics, val_dir))
	plt.close()

	############################# Refit with tuned parameters
	imax = np.argmax(mean_val_score)
	params = {'C': C[imax]}
	np.random.seed(0)
	best_preprocessor = Pipeline([
		('count', CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)),
		('lda', LDAVectorizer(num_topics=num_topics, V=V, 
					alpha=best_lda_params['alpha'],
					kappa=kappa, tau0=tau0, var_i=var_i, 
					size=best_lda_params['size'], perplexity=False))
	])
	best_preprocessor.fit(train_data)
	save_pickle(best_preprocessor, 'result/lda/%d/pre.pkl' % (num_topics))
	best_preprocessor = load_pickle('result/lda/%d/pre.pkl' % (num_topics))
	train_features = best_preprocessor.transform(train_data)
	
	# Best classifier
	best_clf = LinearSVC(max_iter=500, loss='hinge')
	best_clf.set_params(**params)
	best_clf.fit(train_features, train_target)
	save_pickle(best_clf, 'result/svm-lda/%d/clf.pkl' % (num_topics))
	best_clf = load_pickle('result/svm-lda/%d/clf.pkl' % (num_topics))

	############################# Top words
	vocab = best_preprocessor.get_params()['count'].vocabulary_
	inverse_vocab = {}
	for k in vocab.keys():
		inverse_vocab[vocab[k]] = k
	lda_model = best_preprocessor.get_params()['lda'].lda_model	
	top_idxs = lda_model.get_top_words_indexes()
	with open('result/svm-lda/%d/top-words.txt' % num_topics, 'w') as f:
		for i in range(len(top_idxs)):
			s = '\nTopic %d:' % i 
			for idx in top_idxs[i]:
				s += ' %s' % inverse_vocab[idx]
			f.write(s)

	############################ Test
	print('----------- Test')
	test = fetch_20newsgroups(subset='test')
	test_data = load_pickle('dataset/test-data.pkl')[:]
	test_target = test.target[:]
	test_features = best_preprocessor.transform(test_data)
	t0 = time()
	test_pred = best_clf.predict(test_features)
	predict_time = time() - t0
	with open('result/svm-lda/%d/report' % num_topics, 'w') as f:
		f.write('Best preprocessor:\n')
		f.write(str(best_preprocessor.get_params()))
		f.write('\n\n\n')
		f.write('Best classifier:\n')
		f.write(str(best_clf.get_params()))
		f.write('\n\n\n')
		f.write(classification_report(test_target, test_pred))
		f.write('\n\n\n')
		f.write('Predict time: %f' % predict_time)
	############################ Learning curve
	n_train = len(train_target)
	percent = [.2, .4, .6, .8, 1.]
	pool = Pool(processes=3)
	works = []
	for r in percent:
		np.random.seed(0)
		best_preprocessor = Pipeline([
			('count', CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)),
			('lda', LDAVectorizer(num_topics=num_topics, V=V, 
						alpha=best_lda_params['alpha'],
						kappa=kappa, tau0=tau0, var_i=var_i, 
						size=best_lda_params['size'], perplexity=False))
		])
		best_clf = load_pickle('result/svm-lda/%d/clf.pkl' % (num_topics))
		clf_params = best_clf.get_params()
		best_clf = LinearSVC(**clf_params)

		n = int(r * n_train)
		works.append((learning, (best_preprocessor, best_clf, train_data[:n], train_target[:n], \
					test_data, test_target)))

	f1 = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()
	save_pickle((percent, f1), 'result/svm-lda/%d/learning' % num_topics)
	# Plot
	plt.xlabel('Part of train data')
	plt.ylabel('F1 score')
	plt.plot(percent, f1, c='r')
	plt.legend()
	plt.savefig('result/svm-lda/%d/learning_curve.png' % num_topics)
	plt.close()


