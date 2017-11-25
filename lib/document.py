import numpy as np 

class Document:
	def __init__(self, num_terms, num_words, terms, counts):
		self.num_terms = num_terms;
		self.num_words = num_words;
		self.terms = terms;
		self.counts = counts;

	def to_vector(self):
		vec = []
		for i in range(self.num_terms):
			for j in range(self.counts[i]):
				vec.append(self.terms[i])
		return np.array(vec)