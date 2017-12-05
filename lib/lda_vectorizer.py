import numpy as np 
from online_lda import OnlineLDAVB
from document import Document
import math
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from document import count_matrix_to_documents

class LDAVectorizer(BaseEstimator):
	def __init__(self, V, lda_model=None, num_topics=50, alpha=.7,\
				kappa=0.5, tau0=64, var_i=100, size=200, perplexity=True):
		if lda_model:
			self.lda_model = lda_model
		else:
			self.lda_model = OnlineLDAVB(alpha=alpha, K=num_topics, V=V, kappa=kappa, tau0=tau0,\
				batch_size=size, var_max_iter=var_i)	
		self.perplexity = perplexity

	def fit(self, count_matrix, y=None):
		X = count_matrix_to_documents(count_matrix)
		batch_size = self.lda_model.batch_size
		N = len(X)
		np.random.seed(0)
		ids = np.random.permutation(N)
		batchs = range(int(math.ceil(N/float(batch_size))))	
		for i in batchs:
			print('-----LDA minibatch %d' % i)
			batch_ids = ids[i * batch_size: (i + 1) * batch_size]
			t0 = time()
			self.lda_model.fit(X, batch_ids)
			print('-----Minibatch time: %.3f' % (time() - t0))
		return X

	def transform(self, count_matrix):
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

