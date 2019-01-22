from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, \
		ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from random import randint, choice, shuffle, sample
from sys import argv
from data import size, max_len, category_data, EMBEDDING_DIM, enc_gen
from utils import *

batch_size = 1024

def pop_layer(model):
	if not model.outputs:
		raise Exception('Sequential model cannot be popped: model is empty.')

	model.layers.pop()
	if not model.layers:
		model.outputs = []
		model.inbound_nodes = []
		model.outbound_nodes = []
	else:
		model.layers[-1].outbound_nodes = []
		model.outputs = [model.layers[-1].output]
	model.built = False
	# print 'Last layer is now: ' + model.layers[-1].name
	return model

def init_model(preload=None, declare=True, data=None):
	print 'Compiling model...'
	if not declare and preload: return load_model(preload)

	else:
		embedding_matrix, len_word_index = data
		sequence_input = Input(shape=(max_len,), name='sequences',dtype='int32')
		embedded_sequences = Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							name='word_vectors',
							input_length=max_len,
							weights=[embedding_matrix],
							trainable=True)(sequence_input)
		dropped = Dropout(0.1, name='input_dropout') (embedded_sequences)
		x = Convolution1D(16, 3, border_mode='same', name='conv1')(dropped)
		x = ELU() (x)
		x = MaxPooling1D(5, name='maxpoolinglarge')(x)
		x = Convolution1D(32, 3, border_mode='same', name='conv2')(x)
		x = ELU() (x)
		x = MaxPooling1D(2, name='maxpoolingsmall')(x)
		x = Convolution1D(64, 3, border_mode='same', name='conv3')(x)
		x = ELU() (x)
		x = GlobalMaxPooling1D(name='globalmaxpooling')(x)
		pred = Dense(1, activation='sigmoid', name='sentiment_class') (x)
		model = Model(sequence_input, pred)

		if preload:
			model.load_weights(preload)
		return model

def get_data(source='movies', target=None):
	data = None
	if '{}{}'.format(source, target) not in listdir('{}/processedData/'.format(ROOT)):
		data = category_data(source, target=target)
		pickle.dump(data, open('{}/processedData/{}{}'.format(ROOT, source, target),'w'))
	else:
		data = pickle.load(open('{}/processedData/{}{}'.format(ROOT, source, target)))
	return data

def runner(epochs):
	ne = epochs
	x_train, y_train, x_val, y_val = [None] * 4
	pairs = [('electronics','movies')]
	for source, target in pairs:
		epochs = ne
		data = get_data(source, target)
		embedding_matrix = data['embedding_matrix']
		len_word_index = data['len_word_index']
		x_train, y_train = data['x_train'], data['y_train']
		x_val, y_val = data['x_val'], data['y_val']
		model = init_model(data=[embedding_matrix, len_word_index])
		model.compile(optimizer='rmsprop', metrics=['acc'], loss='binary_crossentropy')
		val_checkpoint = ModelCheckpoint('bestval.h5','val_acc', 1, True)
		cur_checkpoint = ModelCheckpoint('current.h5')
		print 'Model compiled.'
		model.fit(x_train, y_train, validation_data=(x_val, y_val),
					nb_epoch=epochs,batch_size=batch_size, verbose=1, callbacks=[val_checkpoint])
		print 'Done with {}, {}'.format(source, target)
def main(args):
	mode, src, dest = None, None, None
	if len(args) == 2: mode, preload = args
	elif len(args) == 4: mode, preload, src, dest = args
	else: raise ValueError('Incorrect number of args.')

	if preload == 'none': preload = None
	if mode == 'vis':
		data = get_data()
		model = init_model(data=[data['embedding_matrix'], data['len_word_index']])
		return visualizer(model)
	if mode == 'train':
		return runner(50)
	if mode == 'confusion':
		model = init_model(preload=preload, declare=False)
		data = category_data(src, target=dest)
		y_true = data['y_val']
		y_pred = model.predict(data['x_val'])
		print get_confusion_matrix(y_true, y_pred)
	else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
