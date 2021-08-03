import os
import time
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import cv2
import numpy as np

def read_dataset(dataset_file, label_file):
	with open(dataset_file, 'rb') as f:
		dataset = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
		dataset = np.reshape(dataset, (-1, 28*28))
	with open(label_file, 'rb') as f:
		labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
	return dataset, labels

def main():
	train_images , train_labels  = read_dataset('./MNIST_data/train-images-idx3-ubyte', './MNIST_data/train-labels-idx1-ubyte') # Training dataset
	test_images, test_labels = read_dataset('./MNIST_data/t10k-images-idx3-ubyte', './MNIST_data/t10k-labels-idx1-ubyte') # Validation dataset

	'''
	# Inspect integrity of training images and labels
	for i in range(20):
		img = train_images[i]
		img = img.reshape(28,28,1)
		print(train_labels[i])
		cv2.imshow('a', img)
		cv2.waitKey(0)
	'''

	# Normalize dataset (data range = 0.0-1.0)
	train_images = train_images.astype(np.float32) / 255.0
	test_images  = test_images.astype(np.float32) / 255.0

	# Convert the class label to one-hot vector   e.g. 3->[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	train_labels_v = keras.utils.to_categorical(train_labels, 10)
	test_labels_v  = keras.utils.to_categorical(test_labels , 10)

	# Build MNIST DL model
	model = keras.Sequential([
		keras.layers.Dense( 128, input_shape=(784,), activation='relu'),
		keras.layers.Dense(10)
	])

	# Compiling the model
	model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'acc'])

	# Training the model
	print('\n*** Start training...')
	model.fit(train_images, train_labels_v, epochs=50)

	# Validate the model
	print('\n*** Start validation...')
	test_loss, test_mae, test_acc = model.evaluate(test_images, test_labels_v)
	print('\nTest accuracy:', test_acc)


	# Saving entire model data (model+weight) in Keras h5 format
	fn_h5 = 'mnist.h5'
	model.save(fn_h5)
	print('*** Keras model data saved : ', fn_h5)

	# Saving weight data of the model in TF checkpoint format
	fn_ckpt = './checkpoint/mnist'
	model.save_weights(fn_ckpt)
	print('*** Checkpoint saved : ', fn_ckpt)

	# Saving entire model data (model+weight) in TF SavedModel format
	fn_savedmodel='./savedmodel'
	if os.path.exists(fn_savedmodel):
		shutil.rmtree(fn_savedmodel)
		print('*** Existing directory {} has been deleted'.format(fn_savedmodel))
		time.sleep(1)     # 'permission denied' error may happen when this sleep(1) is not here. reason=unknown
	keras.experimental.export_saved_model(model, fn_savedmodel)
	print('*** TF SavedModel saved :', fn_savedmodel)

	# Check output node names
	fn_fzpb = 'mnist-frozen.pb'
	num_output = len(model.output_names)
	out_nodes = [ model.outputs[i].name[:model.outputs[i].name.find(':')] for i in range(num_output)]
	print('*** Output node names :',out_nodes)
	# Obtain TF session, replace variables with constant, and save the frozen TF model in protocol buffer format (.pb)
	TFsess = keras.backend.get_session()
	frozen_graph = graph_util.convert_variables_to_constants(TFsess, TFsess.graph.as_graph_def(), out_nodes)
	graph_io.write_graph(frozen_graph, '.', fn_fzpb, as_text=False)
	print('*** {} saved'.format(fn_fzpb))

if __name__ == '__main__':
	main()
	print('\n\n*** Training completed.\n\n')
