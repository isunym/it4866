from nltk.stem.snowball import EnglishStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

############################# Preprocess 
analyzer = CountVectorizer().build_analyzer()
stemmer = EnglishStemmer()

def preprocessor(doc):
	arr = doc.split('\n')
	arr = arr[1:4] + arr[5:6] + arr[8:]
	s = '\n'.join(arr)

	terms = analyzer(doc)
	reg = re.compile('^[a-z]+(\s[a-z]+)*$', re.I)
	filtered = filter(reg.search, terms)

	return ' '.join([stemmer.stem(w) for w in filtered])