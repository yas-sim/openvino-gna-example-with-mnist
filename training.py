import os
import time
import shutil

from keras.datasets import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import cv2
import numpy as np

def main():
	# Prepare datasets (using keras.datasets.mnist)
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape(-1, 28*28)
	test_images = test_images.reshape(-1, 28*28)
	train_labels = train_labels.astype(np.float32)
	test_labels = test_labels.astype(np.float32)

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
	fn_savedmodel='savedmodel'
	if os.path.exists(fn_savedmodel):
		shutil.rmtree(fn_savedmodel)
		print('*** Existing directory {} has been deleted'.format(fn_savedmodel))
		time.sleep(1)     # 'permission denied' error may happen when this sleep(1) is not here. reason=unknown
	model.save(fn_savedmodel, save_format='tf')
	print('*** TF SavedModel saved :', fn_savedmodel)

if __name__ == '__main__':
	main()
	print('\n\n*** Training completed.\n\n')
