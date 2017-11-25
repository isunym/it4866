from pprint import pprint
from time import time
import logging
import numpy as np 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print('Loading 20newsgroup dataset for all categories')

# Load train data
train = fetch_20newsgroups(subset='train')
print('Train data:\n')
print('%d documents' % len(train.filenames))
print('%d categories' % len(train.target_names))

# print(train.target_names[0])
# print(np.where(train.target == 0))

train_data = train.data
train_target = train.target

# print(train_target)
# print(train.filenames)

# Preprocess text
def preprocessor(doc):
	arr = doc.split('\n')
	
	# print(arr)
	
	arr = arr[1:4] + arr[5:6] + arr[8:]
	
	# print('\n'.join(arr))

	return '\n'.join(arr)

# preprocessor(train_data[2])

# K-folds cross validation
if 0:
	skf = StratifiedKFold(n_splits=3)
	results = []
	for train_ids, val_ids in skf.split(train_data, train_target):
		# Data
		training_data = []
		training_labels = []
		for i in train_ids:
			training_data.append(train_data[i])
			training_labels.append(train_target[i])
		validation_data = []
		validation_labels = []
		for i in val_ids:
			validation_data.append(train_data[i])
			validation_labels.append(train_target[i])

		# Params	
		result = {}	
		for C in [1., .1, 0.01]:
			for max_iter in [50, 100, 500]:
				params = 'C%f-max_iter%d' % (C, max_iter)
				
				# Preprocessing and classification	
				pipeline = Pipeline([
					('count', CountVectorizer(stop_words='english', preprocessor=preprocessor, max_df='.9', ngram_range=(1, 1))),
					('tfidf', TfidfTransformer()),
					('clf', LinearSVC(C=C, max_iter=max_iter))
				])

				pipeline.fit(training_data, training_labels)
				predict = pipeline.predict(validation_data)
				
				# print(classification_report(validation_labels, predict))
				
				f1 = f1_score(validation_labels, predict, average='macro')
				result[params] = f1

		results.append(result)

	print(results)
	score_matrix = []
	imax = -1
	for result in results:
		score_matrix.append(result.values())
	imax = np.argmax(np.sum(score_matrix, axis=0))
	print('Best set of parameters:\n')
	print(results[0].keys()[imax])


if 1:
	# Re-fit 
	pipeline = Pipeline([
		('count', CountVectorizer(stop_words='english', preprocessor=preprocessor, max_df=.9, ngram_range=(1, 2))),
		('tfidf', TfidfTransformer()),
		# ('clf', LinearSVC(C=1., max_iter=100))
		# ('clf', MultinomialNB())
		('clf', MLPClassifier())
	])

	pipeline.fit(train_data, train_target)

	# Load test data
	test = fetch_20newsgroups(subset='test')
	test_data = test.data
	test_target = test.target

	test_pred = pipeline.predict(test_data)
	print(classification_report(test_target, test_pred))