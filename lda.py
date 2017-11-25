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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from nltk.stem.snowball import EnglishStemmer
import pickle
from lib.document import Document
import os

############################# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print('Loading 20newsgroup dataset for all categories')

############################# Load train data
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


############################# Preprocess text
def preprocessor(doc):
	arr = doc.split('\n')
	# print(arr)
	arr = arr[1:4] + arr[5:6] + arr[8:]
	# print('\n'.join(arr))
	return '\n'.join(arr)

############################# Stem
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

############################# Pickle
def load_pickle(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

############################# Count vectorizer

count_vect = CountVectorizer(stop_words='english', preprocessor=preprocessor, 
				max_df=1., ngram_range=(1, 2), analyzer=stemmed_words, max_features=20000)

count_vect.fit(train_data)
train_features = count_vect.transform(train_data)

save_pickle(train_features, 'data/train-features.pkl')
save_pickle(count_vect.vocabulary_, 'data/train-vocab.pkl')
save_pickle(count_vect, 'data/count-vectorizer.pkl')


############################# Data for LDA
train_vocab = load_pickle('data/train-vocab.pkl')
train_features = load_pickle('data/train-features.pkl')

V = len(train_vocab)
D_train = train_features.shape[0]

############################# Sparse document to type Document
def to_documents(count_matrix):
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

train_docs = to_documents(train_features)

############################# LDA model
num_topics = 50
alpha = 0.1
kappa = 0.5
tau0 = 64
var_i = 100
size = 500

lda_model = OnlineLDAVB(alpha=alpha, K=num_topics, V=V, kappa=kappa, tau0=tau0,\
				batch_size=size, var_max_iter=var_i)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

result_dir = 'data/lda/var%d-batchsize%d-topics%d-alpha%.2f-kappa%.2f-tau0%d/' \
			% (var_i, size, num_topics, alpha, kappa, tau0)
make_dir(result_dir) 	


############################# Fit minibatchs
ids = range(D_train)
batch_size = lda_model.batch_size
batchs = range(int(math.ceil(D_train/float(batch_size))))	
for i in batchs:
	print('-----LDA minibatch %d' % i)
	batch_ids = ids[i * batch_size: (i + 1) * batch_size]
	t0 = time()
	lda_model.fit(train_docs, batch_ids)
	print('-----Minibatch time: %.3f' % (time() - t0))

save_pickle(lda_model, '%smodel.pkl' % result_dir)

lda_model = load_pickle('%smodel.pkl' % result_dir)

############################# Infer train documents

_, gamma_train = lda_model.infer(train_docs, D_train)
print('Infer train documents: Done')

save_pickle(gamma_train, '%sgamma_train.pkl' % result_dir)

gamma_train = load_pickle('%sgamma_train.pkl' % result_dir) # DxK

############################# Class - topic matrix
num_classes = len(train.target_names)
class_topic = np.zeros((num_classes, num_topics))

for i in range(D_train):
	class_topic[train_target[i], :] += gamma_train[i]

class_topic = 1. * class_topic / np.sum(class_topic, axis=1).reshape(num_classes, 1) # CxK
save_pickle(class_topic, '%sclass-topic.pkl' % result_dir)

class_topic = load_pickle('%sclass-topic.pkl' % result_dir)
# Train predict
train_score = np.dot(gamma_train, class_topic.T) # DxC
train_predict = np.argmax(train_score, axis=1)

print(classification_report(train_target, train_predict))

############################# Load test data
test = fetch_20newsgroups(subset='test')
test_data = test.data
test_target = test.target
D_test = len(test_target)

count_vect = load_pickle('data/count-vectorizer.pkl')
test_features = count_vect.transform(test_data)
save_pickle(test_features, 'data/test-features.pkl')

test_features = load_pickle('data/test-features.pkl')

############################# Predict test
train_docs = to_documents(test_features)

_, gamma_test = lda_model.infer(test_docs, D_test)
print('Infer test documents: Done')

save_pickle(gamma_test, '%sgamma_test.pkl' % result_dir)
gamma_test = load_pickle('%sgamma_test.pkl' % result_dir) # DxK

test_score = np.dot(gamma_test, class_topic.T) # DxC
test_predict = np.argmax(test_score, axis=1)

print(classification_report(test_target, test_predict))