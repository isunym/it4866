import numpy as np 
from online_lda import OnlineLDAVB
from document import Document
import math
from time import time
from sklearn.base import BaseEstimator, TransformerMixin

class LDAClassifier(BaseEstimator):
	def __init__(self, num_topics, num_classes, learning_rate=0.1, max_iter=500, batch_size=50):
		self.num_topics = num_topics # Number of LDA topics
		self.num_classes = num_classes # Number of classes to classify
		self.learning_rate = learning_rate # Learning rate (gradient descent)
		self.max_iter = max_iter # Max iterators (gradient descent)
		self.batch_size = batch_size # Minibatch size (gradient descent)
	
	def fit(self, gamma, y, docs_num_words):
		"""
		Fit classifier with topics infered from documents by LDA model
		Parameters:
			gamma (DxK): topics infered from documents
			y (D): class labels of documents
			docs_num_words (D): number of words in each documents
		"""
		
		# Classes
		classes = {} # map input classes to zero-start classes
		count = 0 # class counter
		for label in y:
			if not classes.has_key(label):
				classes[y] = count
				count += 1
		self.classes = classes

		################################## Class-topic matrix
		D = len(gamma) # number of documents
		E_theta = 1. * gamma / np.sum(gamma, axis=1).reshape(D, 1) # expectation of document topics 
		class_topic = np.zeros((self.num_classes, self.num_topics)) # CxK, distribution on topics of classes

		for d in range(D):
			c = self.classes[y[d]] # class label
			class_topic[c, :] += docs_num_words[d] * E_theta[d]
		# Normalize	
		self.class_topic = 1. * class_topic / np.sum(class_topic, axis=1).reshape(self.num_classes, 1)
		
		################################## Fit class matrix with learing example
		# batchs = int(math.ceil(N / float(batch_size)))
		# i = 0
		# while done == False:
		# 	ids = np.random.permutation(N)
		# 	for j in batchs:
		# 		# Check max iter
		# 		i += 1
		# 		if i > self.max_iter:
		# 			done = True
		# 			break

		# 		# Fit minibatch
		# 		batch_ids = ids[j * self.batch_size: (j + 1) * self.batch_size]
		# 		gamma_mb = gamma[batch_ids]
		# 		y_mb = y[batch_ids]


	def predict(self, gamma):
		score = np.dot(gamma, self.class_topic.T) # DxK . KxC = DxC
		pred = np.argmax(score, axis=1)
		return pred

	def get_params(self, deep=False):
		return {
			'num_topics': self.num_topics, 
			'num_classes': self.num_classes, 
			'learning_rate': self.learning_rate, 
			'max_iter': self.max_iter, 
			'batch_size': self.batch_size
		}	

