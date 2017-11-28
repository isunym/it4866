from pprint import pprint
from time import time
import logging
import numpy as np 
import sys
import math
sys.path.append('../onlinelda')
from model.online_lda import OnlineLDAVB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, f1_score
from util import load_pickle, save_pickle, make_dir
from sklearn.model_selection import learning_curve, validation_curve,\
			cross_val_score, GridSearchCV
import matplotlib.pyplot as plt  
import json
from lib.document import Document 
from lib.lda_vectorizer import LDAVectorizer, count_matrix_to_documents
from multiprocessing import Pool

############################# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def run_lda(preprocess, data, result_dir):
	lda = preprocess.get_params()['lda__lda_model']
	preprocess.fit(data)
	gamma, perplexity = preprocess.transform(data)
	return perplexity

def multi_run_wrapper(args):
   return run_lda(*args)

if __name__ == '__main__': 
	print('Loading 20newsgroup dataset for all categories')

	############################# Load train data
	train = fetch_20newsgroups(subset='train')
	print('Train data:\n')
	print('%d documents' % len(train.filenames))
	print('%d categories' % len(train.target_names))

	train_data = load_pickle('dataset/train-data.pkl')
	train_target = train.target
	D_train = len(train_target)

	############################# Vectorizer
	vect_result = 'max_df=.75, ngram_range=(1, 1), max_features=30000'

	############################# Tune LDA
	V = 30000
	kappa = 0.5
	tau0 = 64
	var_i = 100
	num_topics = 50

	pool = Pool(processes=3)
	works = []
	params = []
	for size in [128, 256, 512]:
		for alpha in [.1, .3, .4, .5, .6, .7, .8, .9]:

			result_dir = 'data/%s/lda/var%d-batchsize%d-topics%d-alpha%.2f-kappa%.2f-tau0%d/' \
						% (vect_result, var_i, size, num_topics, alpha, kappa, tau0)
			make_dir(result_dir) 

			preprocess = Pipeline([
				('count', CountVectorizer(stop_words='english', 
								max_df=.75, ngram_range=(1, 1), max_features=30000)),
				('lda', LDAVectorizer(num_topics=num_topics, V=V, alpha=alpha,
								kappa=kappa, tau0=tau0, var_i=var_i, size=size))
			])

			works.append((preprocess, train_data, result_dir))
			params.append({ 'num_topics': num_topics, 'size': size, 'alpha': alpha})

	perplexities = pool.map(multi_run_wrapper, works)
	pool.close()
	pool.join()
	with open('result/svm-lda/lda-tune', 'w') as f:
		f.write(str(params))
		f.write(str(perplexities))
	save_pickle((params, perplexities), 'result/svm-lda/lda-tune.pkl')	

	############################ Best preprocessor
	params, perplexities = load_pickle('result/svm-lda/lda-tune.pkl')
	imin = np.argmin(perplexities)
	best_lda_params = params[imin]

	# best_lda_params = { 'num_topics': num_topics, 'size': 512, 'alpha': .1}

	# best_preprocessor = Pipeline([
	# 	('count', CountVectorizer(stop_words='english', 
	# 					max_df=.75, ngram_range=(1, 1), max_features=30000)),
	# 	('lda', LDAVectorizer(num_topics=best_lda_params['num_topics'], V=V, 
	# 				alpha=best_lda_params['alpha'],
	# 				kappa=kappa, tau0=tau0, var_i=var_i, 
	# 				size=best_lda_params['size'], perplexity=False))
	# ])


	# ############################ Cross validation
	# # C = np.linspace(1, 3, 10)
	# C = np.logspace(-5, 2, 5)

	# clf = LinearSVC(max_iter=500)
	# estimator = Pipeline([
	# 	('pre', best_preprocessor),
	# 	('clf', clf) 
	# ])
	# parameters = {
	#     'clf__C': C
	# }
	# grid_search = GridSearchCV(estimator, parameters, n_jobs=3, verbose=1)
	# grid_search.fit(train_data, train_target)
	# print(grid_search.cv_results_)
	# mean_train_score = grid_search.cv_results_['mean_train_score']
	# mean_val_score = grid_search.cv_results_['mean_test_score']
	# # Plot
	# plt.xlabel('C')
	# plt.ylabel('Score')
	# plt.plot(C, mean_train_score, c='r', label='Training score')
	# plt.plot(C, mean_val_score, c='b', label='Validation score')
	# plt.legend()
	# plt.savefig('val_curve.png')
	# plt.show()

	# best_estimator = grid_search.best_estimator_ 
	# save_pickle(best_estimator, 'result/svm-lda/estimator.pkl')

	# # ############################# Test
	# best_estimator = load_pickle('result/svm-lda/estimator.pkl')

	# # Load test data
	# test = fetch_20newsgroups(subset='test')
	# # test_data = [preprocessor(doc) for doc in test.data]
	# # save_pickle(test_data, 'dataset/test-data.pkl')
	# test_data = load_pickle('dataset/test-data.pkl')
	# test_target = test.target

	# test_pred = best_estimator.predict(test_data)
	# with open('result/svm-lda/report', 'w') as f:
	# 	f.write('Best estimator:\n')
	# 	f.write(str(best_estimator.get_params()))
	# 	f.write('\n\n\n')
	# 	f.write(classification_report(test_target, test_pred))

	# # Learning curve
	# print(best_estimator.get_params()['pre'].get_params()['count'].vocabulary_)
	# n_train = len(train_target)
	# f1 = []
	# percent = np.linspace(0.1, 1, 10)
	# for r in percent:
	# 	n = int(r * n_train)
	# 	best_estimator.fit(train_data[:n], train_target[:n])

	# 	test_pred = best_estimator.predict(test_data)
	# 	f1.append(f1_score(test_target, test_pred, average='macro'))
	# save_pickle((percent, f1), 'result/svm-lda/learning')
	# # Plot
	# plt.xlabel('Percent of train data')
	# plt.ylabel('F1 score')
	# plt.plot(percent, f1, c='r', label='Test score')
	# plt.legend()
	# plt.savefig('result/svm-lda/learning_curve.png')
	# plt.show()
