from random import randint
import numpy as np
from os.path import dirname, abspath
from os import listdir, remove
from initmodel import size
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gzip

ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/data/'
max_len = 200
EMBEDDING_DIM = 50
max_nb_reviews = 200000
MAX_NB_WORDS = 80000

def read_file(fname):
	texts = []
	labels = []
	pos, neg = 0, 0
	with gzip.open(fname, 'r') as g:
		for line in g:
			review_data = eval(line)
			label = int(review_data['overall'])
			if label == 3: continue
			else: label = float(label//3)
			if label == 1. and pos > neg + 1: continue
			if label == 0. and neg > pos + 1: continue
			text = review_data['reviewText'].strip()
			texts.append(text)
			labels.append(label)
			if labels[-1] == 1.: pos += 1
			elif labels[-1] == 0.: neg += 1
			if len(labels) == max_nb_reviews: break
	return texts, labels			

def get_embeddings_index():
	with open('../glove/glove.6B.{}d.txt'.format(EMBEDDING_DIM)) as f:
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	print 'Found %s word vectors.' % len(embeddings_index)
	return embeddings_index

def compute_reverse_index(word_index):
	index2word = {}
	for word, i in word_index.items():
		index2word[i] = word
	return index2word

def vectorize(x, embedding_matrix):
	y = np.ndarray((len(x), max_len, EMBEDDING_DIM), dtype=np.float32)
	for i, sentence in enumerate(x):
		for j, index in enumerate(sentence):
			y[i][j] = embedding_matrix[index]

	return y

def category_data(category, target=None):
	path = '{}{}.json.gz'.format(DATA_DIR, category)
	if target:
		target_path = '{}{}.json.gz'.format(DATA_DIR, target)

	tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
	texts, labels = read_file(path)

	if target:
		target_texts, target_labels = read_file(target_path)

	else: target_texts, target_labels = [], []
	tokenizer.fit_on_texts(texts + target_texts)
	sequences = tokenizer.texts_to_sequences(texts + target_texts)
	if target:
		sequences, target_sequences = sequences[:-len(target_texts)], sequences[-len(target_texts):]
	word_index = tokenizer.word_index
	# reverse_index = compute_reverse_index(word_index)

	print 'Found %s unique tokens.' % len(word_index)

	data = pad_sequences(sequences, maxlen=max_len)
	if target:
		target_data = pad_sequences(target_sequences, maxlen=max_len)
		target_labels = np.asarray(target_labels, dtype=np.float32)

	labels = np.asarray(labels, dtype=np.float32)
	print 'Shape of data tensor:', data.shape
	print 'Shape of label tensor:', labels.shape 

	embeddings_index = get_embeddings_index()
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
	del embeddings_index

	if not target:
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		data = data[indices]
		labels = labels[indices]
		nb_validation_samples = int(0.2 * data.shape[0])

		x_train = data[:-nb_validation_samples]
		x_val = data[-nb_validation_samples:]
		y_train = labels[:-nb_validation_samples]
		y_val = labels[-nb_validation_samples:]

	else:
		x_train = data
		y_train = labels
		x_val = target_data
		y_val = target_labels

	return {'embedding_matrix':embedding_matrix,
				'len_word_index':len(word_index),
				'x_train':x_train, 
				'y_train':y_train, 
				'x_val':x_val,
				'y_val':y_val}

def enc_gen(x, embedding_matrix, batch_size):
	mini_batches = [x[i:min(len(x), i + batch_size)] for i in xrange(0, len(x), batch_size)]
	while 1:
		for batch in mini_batches:
			batch_y = vectorize(batch, embedding_matrix)
			yield batch, batch_y

