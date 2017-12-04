from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer

############################# Preprocess 
stop_words = set(stopwords.words('english'))
analyzer = CountVectorizer().build_analyzer()
stemmer = EnglishStemmer()

def preprocessor(doc):
	doc = re.sub(r'^from:.*\n', '', doc, flags=re.IGNORECASE)
	doc = re.sub(r'NNTP-Posting-Host.*\n', '', doc, flags=re.IGNORECASE)
	doc = re.sub(r'X-Newsreader.*\n', '', doc, flags=re.IGNORECASE)
	doc = re.sub(r'distribution.*\n', '', doc, flags=re.IGNORECASE)
	words = analyzer(doc)
	reg = re.compile('^[a-z]+$', re.I)
	filtered = filter(reg.search, words)
	stemmed = [stemmer.stem(t.lower()) for t in filtered if t.lower() not in stop_words]
	return ' '.join(stemmed)