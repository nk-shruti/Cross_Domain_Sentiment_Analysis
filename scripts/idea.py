from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, \
		ZeroPadding2D, UpSampling2D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, RMSprop
from keras import backend as K
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
ROOT = dirname(dirname(abspath(__file__)))

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

def customloss(y_true, y_pred):
	# print K.eval(0*K.binary_crossentropy(y_pred, y_true))
	return K.max(y_true) * K.binary_crossentropy(y_pred, y_true)
	
def init_model(preload=None, declare=True, data=None):
	print 'Compiling model...'
	if not declare and preload: return load_model(preload)

	if False:
		model = load_model('top.h5')
		model = Model(model.input, model.get_layer(name='embedding').output)
		generator = Dense(64, name='generator') (model.output)
		x = ELU(name='generator_activation') (generator)
		mapped = Dense(128, name='mapped') (x)
		activated_map = ELU(name='map_activation') (mapped)
		discriminator = Dense(64, name='discriminator') (activated_map)
		x = ELU(name='discriminator_activation') (discriminator)
		domain_pred = Dense(1, activation='sigmoid', name='domain_pred') (x)
		model = Model(model.input, domain_pred)
		for layer in model.layers:
			if layer.name == 'generator': break
			else: layer.trainable = False
		return model

	else:	
		embedding_matrix, len_word_index = data

		sequence_input = Input(shape=(max_len,), dtype='int32')
		embedded_sequences = Embedding(len_word_index + 1,
								EMBEDDING_DIM,
								input_length=max_len,
								# weights=[embedding_matrix],
								trainable=True)(sequence_input)
		x = Convolution1D(32, 3, border_mode='same')(embedded_sequences)
		x = ELU() (x)
		x = MaxPooling1D(5)(x)
		x = Convolution1D(64, 3, border_mode='same')(x)
		x = ELU() (x)
		x = MaxPooling1D(3)(x)
		x = Convolution1D(128, 3, border_mode='same')(x)
		x = ELU() (x)
		x = MaxPooling1D(2)(x)
		x = Dropout(0.5) (x)
		embedding = Flatten(name='embedding')(x)
		# x = ELU() (x)
		# x = Dropout(0.5) (x)
		src_class = Dense(1, activation='sigmoid', name='classifier') (embedding)

		discriminator = Dense(64, name='discriminator') (embedding)
		x = Dropout(0.5) (discriminator)
		x = ELU() (x)
		domain_pred = Dense(1, activation='sigmoid', name='domain_pred') (x)

		model = Model(sequence_input, [src_class, domain_pred])
		names = [layer.name for layer in model.layers]
		# model.compile(SGD(),loss='binary_crossentropy')
		if preload:
			# m = load_model(preload)
			# for layer in m.layers:
			# 	if layer.name in names:
			# 		model.get_layer(name=layer.name).set_weights(layer.get_weights())
			# 	else:
			# 		print 'Layer {} not found'.format(layer.name)
			model.load_weights(preload)

		return model

def get_data():
	source, target = 'electronics', 'movies'
	data = None
	if '{}{}'.format(source, target) not in listdir('{}/processedData/'.format(ROOT)):
		data = category_data(source, autoencoder=False, target=target)
		pickle.dump(data, open('{}/processedData/{}{}'.format(ROOT, source, target),'w'))
	else:
		data = pickle.load(open('{}/processedData/{}{}'.format(ROOT, source, target)))
	return data

def run(epochs, operation, preload=None):
	data = get_data()

	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	src_train, y_train = data['x_train'], data['y_train']
	x_val, y_val = data['x_val'], data['y_val']

	model = init_model(data=[embedding_matrix, len_word_index], preload=preload)
	rmsprop = RMSprop(clipvalue=0.5)
	model.compile(optimizer=rmsprop, loss='binary_crossentropy')
	print 'Model compiled.'

	X = np.concatenate((src_train, x_val))
	y = np.concatenate((y_train, y_val))
	traingen = classifyTrainGen(X, y, batch_size=classify_batch_sz)


def runner(epochs, operation, preload=None, pretrain=False):
	data = get_data()

	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	src_train, y_train = data['x_train'], data['y_train']
	x_val, y_val = data['x_val'], data['y_val']
	nb_validation_samples = int(0.2 * len(src_train))
	src_val, src_y_val = src_train[-nb_validation_samples:], y_train[-nb_validation_samples:]
	src_train, y_train = src_train[:-nb_validation_samples], y_train[:-nb_validation_samples]
	model = init_model(data=[embedding_matrix, len_word_index], preload=preload)
	rmsprop = RMSprop()
	if pretrain:
		for layer in model.layers:
			if layer.name not in ['discriminator', 'domain_pred']:
				layer.trainable = True
	model.compile(optimizer=rmsprop, loss={'domain_pred':'binary_crossentropy', 'classifier':customloss},
		loss_weights={'domain_pred':5 if not pretrain else 1., 'classifier':0.02 if not pretrain else 0.})
	print 'Model compiled.'

	classify_batch_sz = 1024
	classify_traingen = classifyTrainGen(src_train, y_train, batch_size=classify_batch_sz)
	classify_traingenTarget = classifyTrainGen(x_val, y_val, batch_size=classify_batch_sz, target=True)
	gan_gen = gangen(x_val)

	old_acc = 0
	nb_samples = len(src_train)
	restore = dict()

	for epoch in xrange(epochs):
		print 'Epoch : {}'.format(epoch + 1)
		for i in xrange(0, nb_samples, classify_batch_sz):
			if pretrain:
				X1, y1 = classify_traingen.next()
				X2, y2 = classify_traingenTarget.next()
				X, y = join(classify_traingen.next(), classify_traingenTarget.next())
				domain_loss = model.train_on_batch(X, y)[-1]
				if i % 7 == 0:
					print '{} / {}, DLD : {}'.format(i, nb_samples, domain_loss)
			else:
				restore = dict()
				restore['discriminator'] = model.get_layer(name='discriminator').get_weights()
				restore['domain_pred'] = model.get_layer(name='domain_pred').get_weights()

				X,y = classify_traingen.next()
				classifier_loss, src_domain_loss = model.train_on_batch(X, y)[1:]

				X_,y_ = classify_traingenTarget.next()
				model.get_layer(name='discriminator').set_weights(restore['discriminator'])
				model.get_layer(name='domain_pred').set_weights(restore['domain_pred'])

				target_domain_loss = model.train_on_batch(X_, y_)[-1]
				domain_loss1 = ( src_domain_loss + target_domain_loss ) / 2
				model.get_layer(name='discriminator').set_weights(restore['discriminator'])
				model.get_layer(name='domain_pred').set_weights(restore['domain_pred'])
				# model.get_layer(name='sequences').set_weights(restore['sequences'])

				restore = dict()

				for layer in model.layers:
					if layer.name not in ['discriminator', 'domain_pred']:
						restore[layer.name] = layer.get_weights()

				y['domain_pred'] = np.zeros(len(X))
				X, y = join((X,y),(X_, y_))
				domain_loss2 = model.train_on_batch(X, y)[-1]
				# domain_loss2 = None
				if i % 7 == 0:
					print '{} / {}, CL : {}, DLA : {}, DLD : {}'.format(i, nb_samples, classifier_loss, domain_loss1, 
																	domain_loss2)

				for layer in model.layers:
					if layer.name in restore: 
						layer.set_weights(restore[layer.name])
		if not pretrain:
			print 'Train accuracy: {}'.format(accuracy(model.predict(src_val), src_y_val))
			acc = accuracy(model.predict(x_val), y_val)
			if acc > old_acc:
				dumper(model, 'bestval.h5')
				print 'Acc improved from {} to {}, saving model to bestval.h5'.format(old_acc, acc)
				old_acc = acc
			else: print 'Acc = {}'.format(acc)
		else:
			dumper(model, 'pretrained.h5')
def main(args):
	mode, operation = None, None
	if len(args) == 2: mode, preload = args
	elif len(args) == 3: mode, operation, preload = args
	elif len(args) == 4: mode, preload = args
	else: raise ValueError('Incorrect number of args.')

	if preload == 'none': preload = None
	if mode == 'vis':
		data = get_data()
		model = init_model(data=[data['embedding_matrix'], data['len_word_index']])
		return visualizer(model)
	if mode == 'train':
		return runner(1000, operation, preload)
	if mode == 'confusion':
		model = init_model(preload=preload, declare=False)
		data = get_data()
		y_true = data['y_val']
		y_pred = model.predict(data['x_val'])
		print get_confusion_matrix(y_true, y_pred)
	else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
