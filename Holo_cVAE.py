from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import date
from libs.ops import *
from itertools import cycle
""" --------------------------------------------------------------------------------------------"""


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

""" --------------- DECODER Graph --------------------------------------------------------------"""		
def decoder(z, y, train, N_LAT, N_BATCH,  update_collection=tf.GraphKeys.UPDATE_OPS): ## output an x estimate
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up the decoder graph...")
		print("Latent space x/y-dims " + str( np.sqrt(N_LAT).astype(np.int32)) )
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 4 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 4 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8,8, 8])
		## Now combine with the latent variables
		z = tf.reshape(z, [N_BATCH, np.sqrt(N_LAT).astype(np.int32), np.sqrt(N_LAT).astype(np.int32),1]) # latent space - 64		
		concat = tf.concat([z,c], 3) # concat along thirddimension
		## go through additional conv layers to enforce locality in feedback
		c4 = batch_norm(convLayer(concat, 4, 8,3,1,  update_collection=update_collection, padStr="SAME"), name='bn4', is_training=train) ## 8x8 4 channels
		c4 = tf.nn.relu(c4)

		c5 = batch_norm(convLayer(c4, 5, 4,3,1, update_collection=update_collection, padStr="SAME"), name='bn5', is_training=train) ## 8x8 4 channels
		c5 = tf.nn.relu(c5)
		c5 = tf.reshape(c5, [N_BATCH, 8 * 8 *4])
		## dense layer
		#d1 = batch_norm(denseLayer(c5, 4, 512, update_collection=update_collection), name='bn6', is_training=train)
		#d1 = tf.nn.relu(d1)
				
		#do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(c5, 6, 512, update_collection=update_collection), name='bn7', is_training=train)
		d2 = tf.nn.relu(d2)
		#
		do2 = tf.layers.dropout(d2, rate=0.3, training=train)

		d3 = batch_norm(denseLayer(do2, 7, 256, update_collection=update_collection), name='bn8', is_training=train)
		d3 = tf.nn.tanh(d3)

		## Final conv layer
		d3 = tf.reshape(d3, [N_BATCH, 8,8, 4])
		cf = batch_norm(convLayer(d3, 8, 8,3, 1, update_collection=update_collection,  padStr="SAME"), name='bn9', is_training=train)
		cf = tf.reduce_mean(cf,3)				
		cf = tf.nn.relu(cf) ## output activation

		## Reshape and output
		x_hat = tf.reshape(cf, [N_BATCH, 8, 8]) ## make sure this is correct for addition with X_REAL in interpolate
		return x_hat

""" --------------- ENCODER Graph --------------------------------------------------------------"""		

def encoder(x,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS): ## output some gaussian parameters
	with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up encoder graph...")
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 8 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 8 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8* 8* 8])
		## combine reduced conditional variable with setup input - x
		x = tf.reshape(x, [N_BATCH, 8*8]) # fourier space - 64
		concat = tf.concat([x,c], 1) # concat along channel dimension
		## dense layer
		d1 = batch_norm(denseLayer(concat, 4, 512, update_collection=update_collection), name='bn4', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 256, update_collection=update_collection), name='bn5', is_training=train)
		d2 = tf.nn.relu(d2)
				
		#do2 = tf.layers.dropout(d2, rate=0.2, training=train)		

		d3 = batch_norm(denseLayer(d2, 6, 2*N_LAT, update_collection=update_collection), name='bn6', is_training=train)
		#d3 = tf.nn.tanh(d3) ## [-1,1]^128

		## Reshape and output		
		lat_par = tf.reshape(d3, [N_BATCH, N_LAT, 2]) ## [:, :, 0] -> means, [:,:,1] -> log_sigma
		return lat_par
""" --------------------------------------------------------------------------------------------"""		

def unload_gauss_args(lat):
	mu = tf.squeeze(tf.slice(lat, [0, 0, 0], [-1, -1, 1]))
	log_sigma_sq = tf.squeeze(tf.slice(lat, [0, 0, 1], [-1, -1, 1]))
	return mu, log_sigma_sq

# reparametrization trick
def sample(lat, N_LAT, N_BATCH): 
	mu, log_sigma_sq = unload_gauss_args(lat) # both are [N_BATCH, 64]
	## gauss_args is a 1D-tensor
	eps = tf.random.normal([N_BATCH, N_LAT], mean=0.0, stddev=1.)

	return tf.add(mu, tf.multiply(tf.exp(log_sigma_sq/2), eps)) ## [N_BATCH, 64]

def setup_vae_loss(x, x_hat, lat, BETA, N_SAMPLE, N_EPOCH, N_BATCH):
	""" Calculate loss = reconstruction loss + KL loss for each data in minibatch """
	mu, log_sigma = unload_gauss_args(lat)
	# <log P(x | z, y)>	

	reconstruction_loss = BETA*tf.nn.l2_loss(x-x_hat) #tf.reduce_mean(tf.losses.absolute_difference(x, x_hat)) 
	# KL(Q(z | x, y) || P(z | x)) - in analytical form
	kullback_leibler =  N_SAMPLE/N_BATCH*N_EPOCH* 0.5 * tf.reduce_sum(tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma) ## -KL term

	return reconstruction_loss + kullback_leibler

def plotMatrices(yPredict, y):
	plt.subplot(1, 2, 1)
	plt.imshow(yPredict)
	plt.colorbar()
		
	plt.subplot(1, 2, 2)	
	plt.imshow(y)
	plt.colorbar()
	
	plt.show()

""" ----------- MAIN ---------------------------------------------------------------------------"""		
def main(argv):

	## File paths etc
	path = "C:\\Jannes\\learnSamples\\040319_1W_0001s\\validation"
	outPath = "C:\\Jannes\\learnSamples\\040319_validation\\cVAE"
		
	fourier_folder = "inFourier"
	input_folder = 	"in"
	output_folder = "out"
	minFileNr = 1
	indices = get_file_indices(os.path.join(path, fourier_folder))
	maxFile = len(indices) ## number of samples in data set

	#############################################################################
	restore = True ### Set this True to load model from disk instead of training
	testSet = False
	#############################################################################

	save_name = "HOLOVAE.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta = 1e-4
	N_BATCH = 60
	N_VALID = 100	
	N_REDRAW = 5	
	N_EPOCH = 20
	N_LAT = 64
	BETA = 1.0
	## sample size
	N_SAMPLE = maxFile-N_BATCH
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
	LAT = encoder(X,Y, is_train, N_LAT, N_BATCH)
	Z = sample(LAT, N_LAT, N_BATCH)
	X_HAT = decoder(Z,Y, is_train, N_LAT, N_BATCH) ## GENERATOR GRAPH

	## VALIDATION TENSORS
	Z_VALID = tf.random.normal([N_BATCH, N_LAT], mean=0.0, stddev=1)
	X_HAT_VALID = decoder(Z_VALID, Y, is_train, N_LAT, N_BATCH)

	## Loss functions	
	VAE_loss = setup_vae_loss(X, X_HAT, LAT, BETA, N_SAMPLE, N_EPOCH, N_BATCH)
	VAE_solver = tf.train.RMSPropOptimizer(learning_rate=eta).minimize(VAE_loss)
	X_loss = tf.nn.l2_loss(X-X_HAT)
	# Initializer
	initializer = tf.global_variables_initializer() # get initializer   

	# Saver
	saver = tf.train.Saver()	
	print("Commencing training...")
	if testSet:
    		restore = True
	""" ---- TRAINING ------------"""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())    
		if not restore :
			x_err = []
			vae_err = []
			percent=0
			for j in range(N_EPOCH):   
				for i in range(0, N_SAMPLE, N_BATCH):
					if int(100 * ( (j*N_SAMPLE+i)/float(N_SAMPLE*N_EPOCH))) != percent :
						percent = int( 100 * ((j*N_SAMPLE+ i)/float(N_SAMPLE*N_EPOCH)))
						x = load_fourier(i, N_BATCH)
						y = load_output(i, N_BATCH)
						vae_loss = sess.run(VAE_loss, feed_dict={X:x, Y:y, is_train:False} )
						x_loss = sess.run(X_loss, feed_dict={X:x, Y:y, is_train: False})
						x_err.append(x_loss)
						vae_err.append(vae_loss)
						print(str( percent ) + "%"+ " ## xloss " + str(x_loss) + " ## VAE loss " + str(vae_loss))

					x = load_fourier(i, N_BATCH)
					y = load_output(i, N_BATCH)
					sess.run(VAE_solver, feed_dict={X:x, Y:y, is_train: True})			
		
			plt.figure(figsize=(8, 8))
			plt.plot(np.array(x_err), 'r-')
			plt.plot(np.array(vae_err), 'b-')
			plt.show()
			#### SAVE #########		
			save_path = saver.save(sess, save_string)
			print("Model saved in path: %s" % save_path)
			return 

		#### RESTORE MODEL& apply to validate #####
		elif restore:
			saver.restore(sess, save_string)
			

			#### VALIDATION ########
			for k in range(0, N_VALID):
				testNr = last_index + k
				x = load_fourier(testNr, N_BATCH)
				y = load_output(testNr, N_BATCH)
				for r in range(N_REDRAW):
					fileNr = last_index + k*N_REDRAW + r
					## draw new noise
					x_pred = sess.run(X_HAT_VALID, feed_dict={Y:y, is_train: False})

					## write the matrices to file
					if testSet:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
					else:				
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))

			print("DONE! :)")
if __name__ == "__main__":
	main(sys.argv)