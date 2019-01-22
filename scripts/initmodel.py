from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from os.path import dirname, abspath
from os import listdir, remove

ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/data/training/'
size = 128

def init_model(categories):
	contents = []
	for category in categories:
		category = DATA_DIR + category
		posfname = category + '/positive.review.compressed'
		contents.append(open(posfname).read())
		negfname = category + '/negative.review.compressed'
		contents.append(open(negfname).read())
	contents = ''.join(contents).replace(' . ', ' ')
	with open('tempfile.txt','w') as f:
		f.write(contents)
	sentences = LineSentence('tempfile.txt')
	model = Word2Vec(sentences, workers=5, size=size,
				min_count=30, window=30, sample=1e-3)
	model.init_sims(replace=True)
	model.save('ElecsBooks')
	remove('tempfile.txt')

if __name__ == '__main__':
	include = ['books', 'electronics']
	init_model(include)