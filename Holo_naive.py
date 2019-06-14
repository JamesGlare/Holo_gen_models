from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import date
from libs.ops import *

def get_file_indices(path):
	indices = []
	for root, dirs, files in os.walk(path):
		for name in files:
			name_str = str(name)
			if name_str.find(".txt") != -1:
				indices.append( name_str)
	return indices

def load_files(path, nr, i, indices):
	fileContents = []
	for k in range(nr):
		index = indices[i+k]
		fileContents.append(np.loadtxt(os.path.join(path, index), delimiter='\t', unpack=False))
	return np.array(fileContents)

def check_files(path,nr, i, indices):
	result = True
	for k in range(nr):
		result = result and os.path.isfile(os.path.join(path, indices[i+k]))
	return result

def sample_Z(N_BATCH, N_LAT):
    return np.random.uniform(0, 1, size=[N_BATCH, 1, N_LAT])

def denseLayer(x, nr, NOUT, spec_norm=False, update_collection=tf.GraphKeys.UPDATE_OPS):
	### requires [minBatch, elements] shaped tensor
	with tf.variable_scope("denseLayer_"+str(nr), reuse=tf.AUTO_REUSE) as scope:
		""" Interface
		linear(input_, output_size, name="linear", spectral_normed=False, update_collection=None, stddev=None, bias_start=0.0, with_biases=True,
           	with_w=False) """
		return linear(x, NOUT, name=str(nr), spectral_normed=spec_norm, stddev=0.02, update_collection=update_collection) # [minBatch, NOUT]

def convLayer(x, nr, outChannels, kxy, stride, spec_norm=False, update_collection=tf.GraphKeys.UPDATE_OPS, padStr="VALID"):
	with tf.variable_scope("convLayer_"+str(nr), reuse=tf.AUTO_REUSE) as scope:
		""" interface
		conv2d(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):"""
		
		return conv2d(x, outChannels, k_h=kxy, k_w=kxy, name=str(nr), d_h=stride, d_w=stride, stddev=0.02, spectral_normed=spec_norm, update_collection=update_collection, padding=padStr)  

## format matrix in windows-compatible way
def writeMatrices(baseDir, iterNr, pred_fourier, real_int, real_fourier):

	# build dir paths
	pred_fourier_folder = os.path.join(baseDir, "pred_fourier")
	real_int_folder = os.path.join(baseDir, "real_int")
	real_fourier_folder = os.path.join(baseDir, "real_fourier")

	## if directories do not exist, create them
	if not os.path.exists(pred_fourier_folder):
		os.makedirs(pred_fourier_folder)
	if not os.path.exists(real_int_folder):
		os.makedirs(real_int_folder)
	if not os.path.exists(real_fourier_folder):
		os.makedirs(real_fourier_folder)

	#build file paths
	nr_string = '{0:05d}'.format(iterNr)
	pathName_predFourier = os.path.join(pred_fourier_folder, nr_string+".txt")
	pathName_real_int = os.path.join(real_int_folder, nr_string+".txt")
	pathName_real_fourier= os.path.join(real_fourier_folder, nr_string+".txt")

	# save matrices
	np.savetxt(pathName_predFourier, 100.0*pred_fourier, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_int, 255.0*real_int, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier, 100.0*real_fourier , fmt="%.1f", delimiter='\t', newline='\n')


""" --------------- BACKWARD Graph ----------------------------------------------------------"""		
def backward(Y, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("generator") as scope:
		print("Preparing generator graph...")
		## LAYER 1 - conv for y
		Y = tf.reshape(Y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(Y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 4 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8,8 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8,8,8])
	
		## go through additional conv layers to enforce locality in feedback
		c4 = batch_norm(convLayer(c, 4, 8,3,1,  update_collection=update_collection, padStr="SAME"), name='bn4', is_training=train) ## 8x8 4 channels
		c4 = tf.nn.relu(c4)

		c5 = batch_norm(convLayer(c4, 5, 4,3,1, update_collection=update_collection, padStr="SAME"), name='bn5', is_training=train) ## 8x8 4 channels
		c5 = tf.nn.relu(c5)
		c5 = tf.reshape(c5, [N_BATCH, 8 * 8 *4])

		## dense layer

		d1 = batch_norm(denseLayer(c5, 4, 512, update_collection=update_collection), name='bn6', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 512, update_collection=update_collection), name='bn7', is_training=train)
		d2 = tf.nn.relu(d2)
		#do2 = tf.layers.dropout(d2, rate=0.3, training=train)
		## Final conv layer
		d2 = tf.reshape(d2, [N_BATCH, 8,8, 8])
		c4 = batch_norm(convLayer(d2, 7, 8,3, 1, update_collection=update_collection,  padStr="SAME"), name='bn8', is_training=train)
		c4 = tf.reduce_mean(c4,3)				
		c4 = tf.nn.relu(c4) ## output activation

		## Reshape and output
		x = tf.reshape(c4, [N_BATCH, 8, 8]) ## make sure this is correct for addition with X_REAL in interpolate
		return x

def plotMatrices(yPredict, y):
	plt.subplot(1, 2, 1)
	plt.imshow(yPredict)
	plt.colorbar()
		
	plt.subplot(1, 2, 2)	
	plt.imshow(y)
	plt.colorbar()
	
	plt.show()

	

""" --------------- Main function ------------------------------------------------------------"""	
def main(argv):
	### File paths etc
	path = "C:\\Jannes\\learnSamples\\030619_testSet\\"
	outPath = "C:\\Jannes\\learnSamples\\030619_testSet\\naive"
	
	## Check PATHS
	if not os.path.exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not os.path.exists(outPath):
		print("MODEL/OUTPUT PATH DOESN'T EXIST!")
		sys.exit()

	fourier_folder = "inFourier"
	input_folder = 	"in"
	output_folder = "out"
	minFileNr = 1
	indices = get_file_indices(os.path.join(path, output_folder))
	maxFile = len(indices) ## number of samples in data set

	#############################################################################
	restore = True ### Set this True to load model from disk instead of training
	testSet = True
	#############################################################################

	save_name = "HOLONAIVE.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta = 0.0001
	N_BATCH = 60
	N_VALID = 100	
	N_REDRAW = 1	
	N_EPOCH = 20
	BETA = 0.0
	## sample size
	N_SAMPLE = maxFile - N_BATCH
	last_index  = 0
	print("Data set has length "+str(N_SAMPLE))

	### Define file load functions
	load_fourier = lambda x, nr : 1.0/100*np.squeeze(load_files(os.path.join(path, fourier_folder), nr, minFileNr+ x, indices))
	load_input = lambda x, nr : 1.0/255*np.squeeze(load_files(os.path.join(path, input_folder), nr, minFileNr + x, indices))
	load_output = lambda x, nr: 1.0/255*np.squeeze(load_files(os.path.join(path, output_folder), nr, minFileNr + x, indices))

	""" --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	X = tf.placeholder(dtype=tf.float32, name="X") ## Fourier input
	Y = tf.placeholder(dtype=tf.float32, name="Y")
		
	# ROUTE THE TENSORS
	X_HAT = backward(Y, is_train, N_BATCH)

	# Loss function
	back_loss = tf.nn.l2_loss(X-X_HAT)

	naive_solve = tf.train.AdamOptimizer(learning_rate=eta).minimize(back_loss)
	# Initializer
	initializer = tf.global_variables_initializer() # get initializer   

	# Saver
	saver = tf.train.Saver()	
	print("Commencing training...")
	if testSet:
		restore = True
	""" --------------- Session ---------------------------------------------------------------------------"""	
	with tf.Session() as sess:

		sess.run(initializer)
		
		loss_array = [] # Discriminator loss
		percent = 0

		if not restore :
			for j in range(N_EPOCH):   	
				for i in range(0, N_SAMPLE, N_BATCH):
					## Train generator
					x = load_fourier(i, N_BATCH)
					y = load_output(i, N_BATCH)
					sess.run(naive_solve, feed_dict={X: x, Y: y, is_train: True})
					
					## store the progress
					if int(100 * ( (j*N_SAMPLE+i)/float(N_SAMPLE*N_EPOCH))) != percent :
						percent = int( 100 * ((j*N_SAMPLE+ i)/float(N_SAMPLE*N_EPOCH)))
					
						curr_loss = sess.run(back_loss, feed_dict={X: x, Y: y,  is_train: False})
						loss_array.append(curr_loss)
						print(str(percent) + "% ## loss " + str(curr_loss) )
				
					
			    
			plt.figure(figsize=(8, 8))
			plt.plot(np.array(loss_array), 'r-')
			plt.show()
			#### SAVE #########		
			save_path = saver.save(sess, save_string)
			print("Model saved in path: %s" % save_path)
			return

		#### RESTORE MODEL #####
		elif restore:
			saver.restore(sess, save_string)
			
			for k in range(0, N_VALID):
				testNr = last_index + k
				if not testSet:
					x = load_fourier(testNr, N_BATCH)
				y = load_output(testNr, N_BATCH)
				for r in range(N_REDRAW):
					fileNr = last_index + k*N_REDRAW + r
					## draw new noise	
					x_pred = sess.run(X_HAT, feed_dict={ Y:y, is_train: False}) 

					## write the matrices to file
					if testSet:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
					else:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))

			print("DONE! :)")


if __name__ == "__main__":
	main(sys.argv)
