import numpy as np

def readlines(filename):
	with open(filename, 'r') as f:
		return f.readlines()

def write(s, filename):
	with open(filename, 'w') as f:
		f.write(s)

def read_int_array(filename):
	with open(filename) as f:
		lines =  f.readlines()
		result = []
		for line in lines:
			result.append(int(line.strip()))
		return np.array(result)

def save_int_array(arr, filename):
	with open(filename, 'w') as f:
		for t in arr:
			f.write('%d\n' % (t))
	return arr