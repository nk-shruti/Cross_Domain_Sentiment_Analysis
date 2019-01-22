from os.path import dirname, abspath
from os import listdir
from gensim.parsing import PorterStemmer
import re
global_stemmer = PorterStemmer()

ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/data/training/'
stopwords = open(DATA_DIR + 'stopwords').read().split('\n')[::-1]

def clean(sentence):
	sentence = sentence.lower()
	sentence = re.sub("[^a-z]"," ", sentence)
	words = sentence.split()
	clean_words = []
	for word in words:
		word = word.strip()
		if word in stopwords or len(word) == 1: continue
		if len(word):
			# clean_word = ''.join(chars)
			# word = global_stemmer.stem(word)
			clean_words.append(word)
	
	if not clean_words:
		return ''
	sentence = ' '.join(clean_words)
	return sentence

def process(foldername):
	for fname in ['/positive.review', '/negative.review']:
		with open(DATA_DIR + foldername + fname) as f:
			with open(DATA_DIR + foldername + fname + '.compressed','w') as g:
				inReview = False
				review_sentences = []
				for line in f:
					line = line.strip()
					if line == '<review_text>':
						inReview = True
						review_sentences = []
					elif line == '</review_text>': 
						inReview = False
						g.write(' '.join(review_sentences))
						g.write('\n')
					elif inReview:
						if len(line):
							line = clean(line)
							if len(line): review_sentences.append(line)

if __name__ == '__main__':
	included = ['books', 'electronics', 'kitchen']
	for category in included: process(category)
