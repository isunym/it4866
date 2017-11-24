import sys
sys.path.append('../onlinelda')
sys.path.append('../multisvm-classifier/lib')
from model.online_lda import OnlineLDAVB
from preprocessing.read import read_docs, read_n_docs
from test.test import train_model, evaluate_model, make_result_dir, save_model, save_top_words,\
		save_pickle, load_pickle, make_dir
from sklearn.metrics import f1_score, classification_report
import math
import numpy as np
from multiclass_svm import MultiSVM

datapath = '../dataset/20newsgroup/python/'
result_path = 'result/20newsgroup/'

def main(do_load_lda_model=False, \
		do_train_lda=True, \
		do_save_lda_model=True, \
		save_each_minibatch=False, \
		do_infer_train=True, \
		do_train_svm_model=False, \
		do_save_svm_model=False, \
		save_map_labels=False, \
		load_map_labels=False, \
		do_infer_test=False, \
		do_test_svm=True,\
		do_load_test_predict=False, \
		do_save_test_predict=True, \
		do_evaluate=True):

	V = read_count('%svocab-count.txt' % datapath) # number of terms
	D_train = read_count('%strain-count.txt' % datapath) # number of train documents
	# Params
	k = 500
	alpha = 0.1
	kappa = 0.5
	tau0 = 64
	var_i = 100
	size = 500
	
	# Result directory
	result_dir = '%svar%d-batchsize%d-topics%d-alpha%.2f-kappa%.2f-tau0%d/' \
			% (result_path, var_i, size, k, alpha, kappa, tau0)
	make_dir(result_dir)
	# Model lda
	model_file = '%smodel.pkl' % result_dir
	if do_load_lda_model: 
		ldaModel = load_pickle(model_file)
	else:
		ldaModel = OnlineLDAVB(alpha=alpha, K=k, V=V, kappa=kappa, tau0=tau0,\
				batch_size=size, var_max_iter=var_i)

	# Train lda model
	n = size
	if do_train_lda:
		train_corpus_file = open('%strain-data.txt' % datapath)
		for i in xrange(int(math.ceil(D_train / float(n)))):
			# Read n docs
			W = read_n_docs(train_corpus_file, n)
			# Train
			train_model(ldaModel, W, result_dir, save_each_minibatch)
		train_corpus_file.close()

	# Save lda model
	if do_save_lda_model:
		save_pickle(ldaModel, model_file)

	# Save doc-topics
	if do_infer_train:
		features_train = infer_topics(ldaModel, D_train, '%strain-data.txt' % datapath)
		print('Infer train features: Done')		
		save_pickle(features_train, '%sfeatures-train.pkl' % result_dir)
	else:
		features_train = load_pickle('%sfeatures-train.pkl' % result_dir)
	print('Train features: Done')
	features_train = np.array(features_train)

	# Train multiclass svm model
	reg = .1
	svm_batchsize = 1000
	svm_max_iter = 10000
	learning_rate = 0.001
	momentum = 0.5

	svm_result_dir = '%ssvm-hinge-reg%.2f-batchsize%d-n_iters%d-eta%.3f-momentum%.2f/' % (result_dir, reg, svm_batchsize, \
				svm_max_iter, learning_rate, momentum)
	make_dir(svm_result_dir)
	
	train_labels = read_int_array('%strain-label.txt' % datapath)
	if do_train_svm_model:
		svm_model = MultiSVM(lamda=reg, delta=1., batch_size=100, n_iterators=10000, converged=1e-9,
					learning_rate=.001, momentum=.5)
		svm_model.fit(features_train, train_labels)
		if do_save_svm_model:
			save_pickle(svm_model, '%smodel.pkl' % svm_result_dir)
	else:
		svm_model = load_pickle('%smodel.pkl' % svm_result_dir)

	# Test
	test_labels = read_int_array('%stest-label.txt' % datapath)
	print('Load test labels: Done')
	D_test = read_count('%stest-count.txt' % datapath)
	
	# Infer test features
	if do_infer_test:
		features_test = infer_topics(ldaModel, D_test, '%stest-data.txt' % datapath)
		save_pickle(features_test, '%sfeatures_test.pkl' % result_dir)
	else:
		features_test = load_pickle('%sfeatures_test.pkl' % result_dir)
	print('Test features: Done')	

	# Predict
	if do_test_svm:
		if not do_load_test_predict:
			test_predict = svm_model.predict(features_test)
			if do_save_test_predict:
				save_int_array(test_predict, '%stest-predicts.txt' % svm_result_dir)
				print('Save svm test predict: Done')
		else:
			test_predict = read_int_array('%stest-predicts.txt' % svm_result_dir)
			print('Load svm test predict: Done')

	# Evaluate
	if do_evaluate:
		score = f1_score(test_labels, test_predict, average='macro')
		# score = svm_model.score(features_test, test_labels)
		print('Calculate F1 score: Done')
		if do_test_svm:
			save_score(score, '%sscore.txt' % svm_result_dir)
		else:
			save_score(score, '%sscore.txt' % result_dir)


def read_count(filename):
	with open(filename, 'r') as f:
		arr = f.read().strip().split()
		for x in arr:
			if (is_int(x)):
				return int(x)
		return None

def read_int_array(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		result = []
		for line in lines:
			result.append(int(line.strip()))

def write_int_array(arr, filename):
	with open(filename, 'w') as f:
		for it in arr:
			f.write('%d\n' % it)

def is_int(x):
	try:
		x = int(x)
		x += 1
		return True
	except (TypeError, ValueError) as e:
		return False

def infer_topics(ldaModel, D, corpus_filename):
	corpus_file = open(corpus_filename)
	n = 500
	var_gammas = []
	for i in xrange(int(math.ceil(D / float(n)))):
		# Read n docs
		W = read_n_docs(corpus_file, n)
		# Infer
		phi, var_gamma = ldaModel.infer(W, len(W))
		# Get topics
		for gamma_d in var_gamma:
			var_gammas.append(gamma_d)
	corpus_file.close()
	return var_gammas

main()
