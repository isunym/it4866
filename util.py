import pickle
import os

############################# Pickle
def load_pickle(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def save_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)