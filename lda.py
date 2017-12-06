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
			cross_val_score, GridSearchCV, StratifiedKFold, KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
import json
from lib.document import Document 
from lib.lda_vectorizer import LDAVectorizer
from lib.lda_classifier import LDAClassifier
from multiprocessing import Pool

############################# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def multi_run_wrapper(tup):
   return tup[0](*tup[1])

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

def learning(estimator, train_data, train_target, test_data, test_target):
	estimator.fit(train_data, train_target)
	test_predict = estimator.predict(test_data)
	f1 = f1_score(test_target, test_predict, average='weighted')
	return f1

if __name__ == '__main__':
	print('Loading 20newsgroup dataset for all categories')

	############################# Load train data
	train = fetch_20newsgroups(subset='train')
	print('Train data:\n')
	print('%d documents' % len(train.filenames))
	print('%d categories' % len(train.target_names))

	train_data = load_pickle('dataset/train-data.pkl')[:100]
	train_target = train.target[:100]
	D_train = len(train_target)

	############################# Tune LDA
	V = 1000
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

	############################# Tuned LDA
	num_classes = 20
	lda_vect = LDAVectorizer(num_topics=num_topics, V=V, 
					alpha=best_lda_params['alpha'],
					kappa=kappa, tau0=tau0, var_i=var_i, 
					size=best_lda_params['size'], perplexity=False)
	estimator = Pipeline([
		('count', CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)),
		('clf', LDAClassifier(lda_vect, num_classes))
	])

	make_dir('result/lda/%d' % num_topics)
	estimator.fit(train_data, train_target)
	save_pickle(estimator, 'result/lda/%d/estimator.pkl' % (num_topics))
	estimator = load_pickle('result/lda/%d/estimator.pkl' % (num_topics))

	############################# Top words
	vocab = estimator.get_params()['count'].vocabulary_
	inverse_vocab = {}
	for k in vocab.keys():
		inverse_vocab[vocab[k]] = k
	lda_model = estimator.get_params()['clf'].lda_vectorizer.lda_model	
	top_idxs = lda_model.get_top_words_indexes()
	with open('result/lda/%d/top-words.txt' % num_topics, 'w') as f:
		for i in range(len(top_idxs)):
			s = '\nTopic %d:' % i 
			for idx in top_idxs[i]:
				s += ' %s' % inverse_vocab[idx]
			f.write(s)

	############################# Load test data
	test = fetch_20newsgroups(subset='test')
	test_data = load_pickle('dataset/test-data.pkl')[:30]
	test_target = test.target[:30]
	print(test_data)
	D_test = len(test_target)

	test_predict = estimator.predict(test_data)
	test_score = f1_score(test_target, test_predict, average='weighted')

	with open('result/lda/%d/report' % num_topics, 'w') as f:
		f.write('Estimator:\n')
		f.write(str(estimator.get_params()))
		f.write('\n\n\n')
		f.write(classification_report(test_target, test_predict))

	############################ Learning curve
	n_train = len(train_target)
	percent = [.2, .4, .6, .8, 1.]
	pool = Pool(processes=3)
	works = []
	for r in percent:
		lda_vect = LDAVectorizer(num_topics=num_topics, V=V, 
					alpha=best_lda_params['alpha'],
					kappa=kappa, tau0=tau0, var_i=var_i, 
					size=best_lda_params['size'], perplexity=False)
		estimator = Pipeline([
			('count', CountVectorizer(max_df=.8, min_df=3, ngram_range=(1, 3), max_features=V)),
			('clf', LDAClassifier(lda_vect, num_classes))
		])

		n = int(r * n_train)
		works.append((learning, (estimator, train_data[:n], train_target[:n], \
					test_data, test_target)))
		# learning(estimator, train_data[:n], train_target[:n], \
		# 			test_data, test_target)

	f1 = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()
	save_pickle((percent, f1), 'result/lda/%d/learning' % num_topics)
	# Plot
	plt.xlabel('Part of train data')
	plt.ylabel('F1 score')
	plt.plot(percent, f1, c='r')
	plt.legend()
	plt.savefig('result/lda/%d/learning_curve.png' % num_topics)
	# plt.show()
		