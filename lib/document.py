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