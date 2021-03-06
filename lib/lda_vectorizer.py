import numpy as np 
from online_lda import OnlineLDAVB
from document import Document
import math
from time import time
from sklearn.base import BaseEstimator, TransformerMixin

def count_matrix_to_documents(count_matrix):
	documents = []
	for i in range(count_matrix.shape[0]):
		row = count_matrix[i].toarray()[0]
		pos = np.where(row > 0)[0]
		num_terms = len(pos)
		num_words = np.sum(row[pos])
		terms = pos
		counts = row[pos]	
		documents.append(Document(num_terms, num_words, terms, counts))
	return documents

class LDAVectorizer(BaseEstimator):
	def __init__(self, V, lda_model=None, num_topics=50, alpha=.7,\
				kappa=0.5, tau0=64, var_i=100, size=200, perplexity=True):
		if lda_model:
			self.lda_model = lda_model
		else:
			self.lda_model = OnlineLDAVB(alpha=alpha, K=num_topics, V=V, kappa=kappa, tau0=tau0,\
				batch_size=size, var_max_iter=var_i)	
		self.perplexity = perplexity

	def fit(self, count_matrix, y):
		X = count_matrix_to_documents(count_matrix)
		batch_size = self.lda_model.batch_size
		N = len(X)
		ids = np.random.permutation(N)
		batchs = range(int(math.ceil(N/float(batch_size))))	
		for i in batchs:
			print('-----LDA minibatch %d' % i)
			batch_ids = ids[i * batch_size: (i + 1) * batch_size]
			t0 = time()
			self.lda_model.fit(X, batch_ids)
			print('-----Minibatch time: %.3f' % (time() - t0))
		return self

	def transform(self, count_matrix, perplexity=False):
		X = count_matrix_to_documents(count_matrix)
		phi, gamma = self.lda_model.infer(X, len(X))
		if self.perplexity:
			perplexity = self.lda_model.perplexity(X, phi, gamma)
			return gamma, perplexity
		else:
			return gamma

	def get_params(self, deep):
		return {'lda_model': self.lda_model, 'num_topics': self.lda_model.K, 'alpha': self.lda_model.alpha,\
				'kappa': self.lda_model.kappa, 'tau0': self.lda_model.tau0, \
				'var_i': self.lda_model.var_max_iter, 'size': self.lda_model.batch_size,\
				'V': self.lda_model.V, 'perplexity': self.perplexity}	

