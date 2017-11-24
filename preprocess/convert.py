from fileio import readlines

# Convert data from matlab format to python format
def convert(fromfile, tofile):
	# Read
	lines = readlines('%s' % fromfile)
	docs = {}
	for line in lines:
		arr = line.split()
		docid = arr[0]
		termid = int(arr[1]) - 1
		count = arr[2]
		if termid < 53975:
			if not docs.has_key(docid):
				docs[docid] = [(termid, count)]
			else:	
				docs[docid].append((termid, count))
	# Write
	with open('%s' % tofile, 'w') as f:
		s = ''
		for doc in docs.values(): 
			count = len(doc)			
			s += str(count)	
			for i in range(count):
				s += ' %s:%s' % (doc[i][0], doc[i][1])
			s += '\n'
		f.write(s)

convert('../../dataset/20newsgroup/matlab/test.data', \
		'../../dataset/20newsgroup/python/test-data.txt')