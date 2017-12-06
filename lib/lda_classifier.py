import numpy as np 
from online_lda import OnlineLDAVB
from document import Document
from lda_vectorizer import LDAVectorizer
import math
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse

def convert_labels(y, C):
	"""
	 convert 1d label to a matrix label: each column of this
	 matrix coresponding to 1 element in y. In i-th column of Y,
	 only one non-zeros element located in the y[i]-th position,
	 and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return
	 [[1, 0, 0, 1],
	 [0, 0, 1, 0],
	 [0, 1, 0, 0]]
	 """
	Y = sparse.coo_matrix((np.ones_like(y),
	(y, np.arange(len(y)))), shape=(C, len(y))).toarray()
	return Y

def softmax_stable(Z):
	"""
	 Compute softmax values for each sets of scores in Z.
	 each column of Z is a set of score.
	 """
	e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
	A = e_Z / e_Z.sum(axis = 0)
	return A

class LDAClassifier(BaseEstimator):
	def __init__(self, lda_vectorizer, num_classes, learning_rate=0.01, max_iter=500, batch_size=5, converged=1e-6):
		self.lda_vectorizer = lda_vectorizer
		lda_vectorizer.perplexity = False
		self.num_topics = lda_vectorizer.lda_model.K
		self.num_classes = num_classes # Number of classes to classify
		self.learning_rate = learning_rate # Learning rate (gradient descent)
		self.max_iter = max_iter # Max iterators (gradient descent)
		self.batch_size = batch_size # Minibatch size (gradient descent)
		self.converged = converged

	def fit(self, count_matrix, y):
		"""
		Fit classifier with topics infered from documents by LDA model
		Parameters:
			count_matrix (DxV): word counts matrix
			y (D): class labels of documents
			docs_num_words (D): number of words in each documents
		"""

		# Fit lda vectorizer
		docs = self.lda_vectorizer.fit(count_matrix, y) # documents
		gamma = np.array(self.lda_vectorizer.transform(count_matrix))
		
		# Classes
		# classes = {} # map input classes to zero-start classes
		# count = 0 # class counter
		# for label in y:
		# 	if not classes.has_key(label):
		# 		classes[label] = count
		# 		count += 1
		# self.classes = classes
		# print(y)
		# y = np.vectorize(lambda yi: self.classes[yi])(y)
		# print(y)
		################################## Class-topic matrix
		D = len(gamma) # number of documents
		E_theta = gamma / np.sum(gamma, axis=1).reshape(D, 1) # expectation of document topics 
		class_topic = np.zeros((self.num_classes, self.num_topics)) # CxK, distribution on topics of classes

		for d in range(D):
			# c = self.classes[y[d]] # class label
			c = y[d]
			class_topic[c, :] += docs[d].num_words * np.array(E_theta[d]) 
		# Normalize	
		self.class_topic = 1. * class_topic / np.sum(class_topic, axis=1).reshape(self.num_classes, 1)
		################################## Fit class matrix with learing example
		v = np.zeros_like(self.class_topic)
		batchs = int(math.ceil(D / float(self.batch_size)))
		i = 0
		loss_history = []
		done = False
		while done == False:
			print('-----------------------------------Iterator number %d' % i)
			ids = np.random.permutation(D)
			for j in range(batchs):
				# Check max iter
				i += 1
				if i > self.max_iter:
					done = True
					break

				# Fit minibatch
				batch_ids = ids[j * self.batch_size: (j + 1) * self.batch_size]
				gamma_mb = gamma[batch_ids]
				y_mb = y[batch_ids]

				# Gradient descent
				vt = self.learning_rate * self.gradient(gamma_mb, y_mb)
				W_new = self.class_topic - vt
				# Check convergence
				if np.average((W_new - self.class_topic) ** 2) < self.converged:
					done = True
					break 
				self.class_topic = W_new
				v = vt

				# Loss of minibatch
				loss, _ = self.loss(X_mb, y_mb)
				if i % 10 == 0:
					loss_history.append(loss)
		return loss_history

	def gradient(self, gamma, y):
		grad = (softmax_stable(self.class_topic.dot(gamma.T)) -\
			convert_labels(y, self.num_classes)).dot(gamma)
		return grad

	def loss(self, gamma, y):
		

	def predict(self, count_matrix):
		gamma = self.lda_vectorizer.transform(count_matrix)
		score = np.dot(gamma, self.class_topic.T) # DxK . KxC = DxC
		pred = np.argmax(score, axis=1)
		return pred

	def get_params(self, deep=False):
		return {
			'lda_vectorizer': self.lda_vectorizer, 
			'num_classes': self.num_classes, 
			'learning_rate': self.learning_rate, 
			'max_iter': self.max_iter, 
			'batch_size': self.batch_size
		}	

