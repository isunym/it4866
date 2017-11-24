from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class ToArray(object):
    def transform(self, X):
        return X.toarray()
    def fit(self, X, y=None):
        return self

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print('Loading 20newsgroup dataset for all categories')

train_data = fetch_20newsgroups(subset='train')
print(train_data.data[1])

print('train data:\n')
print('%d documents' % len(train_data.filenames))
print('%d categories' % len(train_data.target_names))

pipeline = Pipeline([
	('tfidf', TfidfVectorizer(max_df=1., max_features=10000, ngram_range=(1, 2),
			norm='l2')),
	('clf', LinearSVC(class_weight=None, dual=True, fit_intercept=True,
				intercept_scaling=1, loss='squared_hinge', max_iter=10000,
				multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
				verbose=0))
])

parameters = {
	'clf__C': (1.,)
}

if __name__ == '__main__':
	
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
	print('Performing grid search...')
	print('pipeline', [name for name in pipeline.steps])
	print('parameters:')
	pprint(parameters)
	t0 = time()
	grid_search.fit(train_data.data, train_data.target)
	print('Best score %0.3f' % grid_search.best_score_)
	print('Best parameters set:')
	best_params = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print('%s: %r' % (param_name, best_params[param_name]))
	

	# Test
	test_data = fetch_20newsgroups(subset='test')
	print('Test data:\n')
	print('%d documents' % len(test_data.filenames))
	print('%d categories' % len(test_data.target_names))
	print(grid_search.score(test_data.data, test_data.target))