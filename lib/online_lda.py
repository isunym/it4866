import numpy as np
from scipy.special import digamma, gamma, gammaln
import math
from scipy.sparse import coo_matrix
import time
from document import Document
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import csv
from sklearn.base import BaseEstimator, TransformerMixin

def normalize(X, axis):
	if axis == 0:
		# Column nomalize
		X1 = 1. * X / np.sum(X, axis=0) 
	elif axis == 1:
		# Row normalize
		X1 = 1.* X / np.sum(X, axis=1).reshape(X.shape[0], 1)
	else:
		return X
	return X1

class OnlineLDAVB(BaseEstimator):
	def __init__(self, alpha=0.1, beta=None, init_beta='random', reinit_beta=False, tau0=None, kappa=None, \
				K=None, V=None, predictive_ratio=.8, \
				var_converged=1e-6, var_max_iter=50, batch_size=100, t=0):
		self.K = K # Number of topics
		self.V = V # Dictionary size
		self.alpha = alpha # Dirichlet parameters of topics distribution 
		# Topic - term probability
		if beta is None:
			self._init_beta_random() 
		else:
			self.beta = beta
		self.kappa = kappa # Control the rate old values of beta are forgotte
		self.tau0 = tau0 # Slow down the early stop iterations of the algorithm
		self.predictive_ratio = predictive_ratio # Predictive observed - held-out ratio
		self.var_converged = var_converged 
		self.var_max_iter = var_max_iter
		self.batch_size = batch_size # Batch size
		self.reinit_beta = reinit_beta
		self.init_beta = init_beta
		self.t = t

	# Get parameters for this estimator.
	def get_params(self, deep=False):
		return {'K': self.K, 'V': self.V, 'alpha': self.alpha, 'tau0': self.tau0, 'kappa': self.kappa, \
				'predictive_ratio': self.predictive_ratio, 'var_converged': self.var_converged, 
				'var_max_iter':self.var_max_iter, \
				'batch_size': self.batch_size, 't': self.t,
				'beta': self.beta}

	# Init beta	
	def _init_beta_random(self):
		# Multinomial parameter beta: KxV
		np.random.seed(0)
		self.beta = normalize(np.random.gamma(100, 1./100, (self.K, self.V)), axis=1)

	# Init beta corpus
	def _init_beta_corpus(self, W, D):
		self.beta = np.zeros((self.K, self.V))
		num_doc_per_topic = 5

		for i in range(num_doc_per_topic):
		    rand_index = np.random.permutation(D)
		    for k in range(self.K):
		        d = rand_index[k]
		        doc = W[d]
		        for n in range(doc.num_terms):
		            self.beta[k][doc.terms[n]] += doc.counts[n]
		self.beta += 1
		self.beta = normalize(self.beta, axis=1)

	def log_info(self, log):
		log.write('---------------------------------\n')
		log.write('Online LDA:\n')
		log.write('Number of topics: %d\n' % self.K)
		log.write('Number of terms: %d\n' % self.V)
		log.write('Batch size: %d\n' % self.batch_size)
		log.write('alpha=%f\n' % self.alpha)
		log.write('tau0=%f\n' % self.tau0)
		log.write('kappa=%f\n' % self.kappa)
		log.write('var_converged=%f\n' % self.var_converged)
		log.write('var_max_iter=%d\n' % self.var_max_iter)
		log.write('----------------------------------\n')

	# Fit data
	def fit_batch(self, W):
		"""
			W: list of documents
		"""
		# Re initialize beta
		if self.init_beta == 'corpus' and self.reinit_beta:
			self._init_beta_corpus(W, len(W))
		# Run EM
		self._em(W)

	# Fit minibatch
	def fit(self, W, batch_ids):
		# Re initialize beta
		if self.init_beta == 'corpus' and self.reinit_beta:
			self._init_beta_corpus(W, len(W))
		# Run EM on minibatch
		self._em_minibatch(W, batch_ids)

	def _em_minibatch(self, W, batch_ids):
		# Estimation for minibatch
		suff_stat = self._estimate(W, batch_ids)
		# Update beta
		beta_star = self._maximize(suff_stat) # intermediate
		ro_t = (self.tau0 + self.t) ** (-self.kappa) # update weight
		self.beta = (1 - ro_t) * self.beta + ro_t * beta_star	
		# Time
		self.t += 1
		
	# EM with all documents
	def _em(self, W):
		D = len(W)
		# Permutation
		random_ids = np.random.permutation(D)
		# For minibatch
		batchs = range(int(math.ceil(D/self.batch_size)))
		for i in batchs:
			# Batch documents id
			batch_ids = random_ids[i * self.batch_size: (i + 1) * self.batch_size]
			# EM minibatch
			self._em_minibatch(W, batch_ids)


	# Init variational parameters for each document
	def _doc_init_params(self, W_d):
		phi_d = np.ones((W_d.num_words, self.K)) / self.K
		gamma_d = (self.alpha + 1. * W_d.num_words / self.K) * np.ones(self.K)	
		return phi_d, gamma_d	

	# Estimate batch
	def _estimate(self, W, batch_ids):
		# Init sufficiency statistic for minibatch
		suff_stat = np.zeros(self.beta.shape)
		# For document in batch
		for d in batch_ids:
			# Estimate doc
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			# Update sufficiency statistic
			for j in range(W[d].num_words):
				for k in range(self.K):
					suff_stat[k][W_d[j]] += phi_d[j][k]
		return suff_stat

	def _estimate_doc(self, W, d):
		# Document flatten
		W_d = W[d].to_vector()	
		# Init variational parameters
		phi_d, gamma_d = self._doc_init_params(W[d])

		# Coordinate ascent
		old_gamma_d = gamma_d
		for i in range(self.var_max_iter):
			# Update phi
			phi_d = normalize(self.beta.T[W_d, :] * np.exp(digamma(gamma_d)), axis=1)
			# Update gamma
			gamma_d = self.alpha + np.sum(phi_d, axis=0)

			# Check convergence
			meanchange = np.mean(np.fabs(old_gamma_d - gamma_d))
			if meanchange < self.var_converged:
				break
			old_gamma_d = gamma_d
		return phi_d, gamma_d, W_d

	# Update global parameter
	def _maximize(self, suff_stat):
		return normalize(suff_stat, axis=1) + 1e-100

	# Get top words of each topics
	def get_top_words_indexes(self):
		top_idxs = []
		# For each topic
		for t in self.beta:
			desc_idx = np.argsort(t)[::-1]
			top_idx = desc_idx[:20]
			top_idxs.append(top_idx)
		return np.array(top_idxs)	

	# Inference new docs
	def infer(self, W, D):
		phi = []
		var_gamma = []
		for d in range(D):
			phi_d, gamma_d, W_d = self._estimate_doc(W, d)
			phi.append(phi_d)
			var_gamma.append(gamma_d)
		return phi, var_gamma	
		
	# Calculate lower bound
	def _lower_bound(self, W, D, phi, var_gamma):
		print 'Compute lower bound'
		result = 0
		t0 = time.time()
		for d in range(D):
			result += self._doc_lower_bound(W, d, phi, var_gamma)
		print "Lower bound time: %f" % (time.time() - t0)
		return result

	# Document lower bound
	def _doc_lower_bound(self, W, d, phi, var_gamma):
		sub_digamma = digamma(var_gamma[d]) - digamma(np.sum(var_gamma[d]))
		# Eq log(P(theta|alpha))
		A1 = gammaln(self.K * self.alpha) - self.K * gammaln(self.alpha)
		A = A1 + np.sum((self.alpha - 1) * sub_digamma) # A = 0
		# SUMn Eq log(P(Zn|theta))
		B = np.sum(phi[d].dot(sub_digamma))
		# SUMn Eq log(P(Wn|Zn, beta))
		C1 = np.nan_to_num(np.log((self.beta[:, W[d].to_vector()]).T)) # NxK
		C = np.sum(phi[d] * C1)
		# Eq log(q(theta|gamma))
		D1 = (var_gamma[d] - 1).dot(sub_digamma) # 1xK . Kx1 = 1
		D2 = gammaln(np.sum(var_gamma[d])) - np.sum(gammaln(var_gamma[d]))
		D = D1 + D2
		# SUMn Eq log(q(Zn))
		E = np.sum(phi[d] * np.nan_to_num(np.log(phi[d])))
		result = A + B + C - D - E
		# print 'Document lower bound time: %f' % (time.time() - start) 
		return result
	
	# Perplexity
	def perplexity(self, W, phi, var_gamma):
		D = len(W) # number of documents
		# Lower bound likelihood
		lower_bound = self._lower_bound(W, D, phi, var_gamma)
		num_words = 0
		for doc in W:
			num_words += np.sum(doc.counts)
		# Perplexity
		perplexity = np.exp(-lower_bound / num_words)
		return perplexity