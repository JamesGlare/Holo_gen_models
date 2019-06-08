from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import date
from libs.ops import *

""" --------------------------------------------------------------------------------------------"""
TWOPI_ = 1.0/np.sqrt(2*np.pi)

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

""" --------------- Graph & Related functions  --------------------------------------------------------------"""	

# reparametrization trick
def sample(PI, SIGMA, MU, N_BATCH, K, L): 
	## decide on which pi to use
	## Option 1 - Use categorical distribution
	#CAT_DIST = tf.distributions.Categorical(probs=PI)
	#k = CAT_DIST.sample(1) ## not quite sure how to deal with Batch redraw here ... for loop?

	## Option 2 - Just take the max component
	k = tf.math.argmax(PI, axis=1) ## [N_BATCH]
	MU_k = tf.squeeze(tf.gather(MU, k, axis=1)) ## [N_BATCH, L]
	### MU_k = tf.reshape(MU[:,k,:], [N_BATCH, L]) ## get the mean [N_BATCH, L]

	## Sample from standard normal distribution
	eps = tf.random.normal([N_BATCH,L], mean=0.0, stddev=1.)
	SIGMA_k = SIGMA[:,k] ## ... and variance [N_BATCH,]
	return tf.add(MU_k, tf.multiply(SIGMA_k, eps)) ## [N_BATCH,L] (broadcasting abuse)


def mixture_density(Y, PI, SIGMA, MU, N_BATCH, K, L):
	## PI -> [N_BATCH, K]
	## SIGMA -> [N_BATCH, K]
	## MU -> [N_BATCH, K, L]
	Y_tile = tf.tile(Y, [K, 1]) 								## [K*N_BATCH, L]
	Y_tile = tf.reshape(Y_tile, [N_BATCH, K, L])
	dYSq = tf.reduce_sum(tf.square(Y-MU), axis=2) 				## [N_BATCH, K] 
	expon = tf.multiply(dYSq, tf.reciprocal( 2*tf.square(SIGMA)))  			## [N_BATCH, K]
	norm =  tf.reciprocal(tf.multiply(tf.sqrt( TWOPI_),SIGMA)) 	## [N_BATCH, K]
	gauss = tf.multiply(norm, tf.exp(-expon)) 										## [N_BATCH, K]
	return tf.squeeze(tf.matmul(PI, tf.transpose(gauss)))		## [N_BATCH]

def MDN(y, train, N_BATCH, K, L,  update_collection=tf.GraphKeys.UPDATE_OPS): ## output an x estimate
	with tf.variable_scope("MDN", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up the MDN graph...")
        
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		### Convolutional layers
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 8 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 8 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8* 8* 8])
        ### End of convolutional layers
		d1 = batch_norm(denseLayer(c, 4, 512, update_collection=update_collection), name='bn4', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 256, update_collection=update_collection), name='bn5', is_training=train)
		d2 = tf.nn.relu(d2)
				
		do2 = tf.layers.dropout(d2, rate=0.2, training=train)		

		d3 = batch_norm(denseLayer(do2, 6, K*(2*L+2), update_collection=update_collection), name='bn6', is_training=train)
        ## Convention pi sigma mu_0, ..., mu_(N-1)
		par = tf.reshape(d3, [N_BATCH, K, 2*L+2])
        
		pi = tf.exp(par[:,:,0]) 						## [N_BATCH, K]
		norm_pi = tf.reciprocal(tf.reduce_sum(pi, 1, keep_dims=True)) ## also [N_BATCH, K]
		pi = tf.multiply(pi, norm_pi)						## [N_BATCH, K]
		sigma = tf.exp(par[:,:,1])						## [N_BATCH, K]
		mu = par[:,:,2:]								## [N_BATCH, K, L]

		return pi, sigma, mu

""" --------------------------------------------------------------------------------------------"""		


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
	
	## File paths etc ############################################################
	path = "C:\\Jannes\\learnSamples\\040319_1W_0001s"
	outPath = "C:\\Jannes\\learnSamples\\040319_validation\\MDN"
	##############################################################################
	## Check PATHS
	if not os.path.exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not os.path.exists(outPath):
		print("MODEL/OUTPUT PATH DOESN'T EXIST!")
		sys.exit()

	save_name = "HOLOMDN.ckpt"
	save_string = os.path.join(outPath, save_name)

	fourier_folder = "inFourier"
	input_folder = 	"in"
	output_folder = "out"

	minFileNr = 1
	indices = get_file_indices(os.path.join(path, fourier_folder))
	maxFile = len(indices) ## number of samples in data set

	#############################################################################
	restore = False ### Set this True to load model from disk instead of training
	testSet = False
	#############################################################################

	### Hyperparameters
	tf.set_random_seed(42)
	eta = 1e-4
	N_BATCH = 60
	N_VALID = 100	
	N_REDRAW = 5	
	N_EPOCH = 30
	K = 10 	### number of peaks
	L = 8	### dimensions of X
	## sample size
	N_SAMPLE = maxFile-N_BATCH
	last_index  = 0
	print("Data set has length "+str(N_SAMPLE))

	### Define file load functions
	load_fourier = lambda x, nr : 1.0/100*np.squeeze(load_files(os.path.join(path, fourier_folder), nr, minFileNr+ x, indices))
	load_input = lambda x, nr : 1.0/255*np.squeeze(load_files(os.path.join(path, input_folder), nr, minFileNr + x, indices))
	load_output = lambda x, nr: 1.0/255*np.squeeze(load_files(os.path.join(path, output_folder), nr, minFileNr + x, indices))
	""" --------------- Set up the graph ---------------------------------------------------------"""	

	## Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	X = tf.placeholder(dtype=tf.float32, name="X") ## Fourier input
	Y = tf.placeholder(dtype=tf.float32, name="Y")
		
	## ROUTE THE TENSORS
	PI, SIGMA, MU = MDN(Y, is_train,N_BATCH,K,L)

	## VALIDATION TENSORS
	X_HAT = sample(PI, SIGMA, MU, N_BATCH, K, L)

	## Loss functions	
	MDN_loss = -tf.reduce_sum(tf.log(mixture_density(Y, PI, SIGMA, MU, N_BATCH, K, L) ) )
	MDN_solver = tf.train.RMSPropOptimizer(learning_rate = eta).minimize(MDN_loss)
	X_loss = tf.nn.l2_loss(X-X_HAT)
	## Initializer
	initializer = tf.global_variables_initializer() # get initializer   

	# Saver
	saver = tf.train.Saver()	
	print("Commencing training...")

	""" ---- TRAINING ------------"""
	with tf.Session() as sess:
		sess.run(initializer)    
		if not restore :
			# setup error
			x_err = []
			mdn_err = []
			percent=0
			for j in range(N_EPOCH):   
				for i in range(0, N_SAMPLE, N_BATCH):
					if int(100 * ( (j*N_SAMPLE+i)/float(N_SAMPLE*N_EPOCH))) != percent :
						percent = int( 100 * ((j*N_SAMPLE+ i)/float(N_SAMPLE*N_EPOCH)))
						x = load_fourier(i, N_BATCH)
						y = load_output(i, N_BATCH)
						mdn_loss = sess.run(MDN_loss, feed_dict={X:x, Y:y, is_train:False} )
						x_loss = sess.run(X_loss, feed_dict={X:x, Y:y, is_train: False})
						x_err.append(x_loss)
						mdn_err.append(mdn_loss)
						print(str( percent ) + "%"+ " ## xloss " + str(x_loss) + " ## MDN loss " + str(mdn_loss))

					x = load_fourier(i, N_BATCH)
					y = load_output(i, N_BATCH)
					sess.run(MDN_solver, feed_dict={X:x, Y:y, is_train: True})			
		
			plt.figure(figsize=(8, 8))
			plt.plot(np.array(x_err), 'r-')
			plt.plot(np.array(mdn_err), 'b-')
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
					x_pred = sess.run(X_HAT, feed_dict={Y:y, is_train: False})

					## write the matrices to file
					if testSet:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
					else:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))
			print("DONE! :)")
if __name__ == "__main__":
	main(sys.argv)
