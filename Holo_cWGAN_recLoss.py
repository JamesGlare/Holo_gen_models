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


""" --------------- Generator Graph ----------------------------------------------------------"""		
def generator(z,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("generator") as scope:
		print("Preparing generator graph...")
		## LAYER 1 - conv for y
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), is_training=train, name="bn1") ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), is_training=train, name="bn2") ## 10x10, 4 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection),  is_training=train, name="bn3") ## 8x8, 4 channels
		c3 = tf.nn.relu(c3)

		c = tf.reshape(c3, [N_BATCH, 8,8,8])
		## Now combine with the latent variables
		z = tf.reshape(z, [N_BATCH, 8,8,1]) # latent space - 64
		concat = tf.concat([z,c], 3) # concat along second dimension
		
		## go through additional conv layers to enforce locality in feedback
		c4 = batch_norm(convLayer(concat, 4, 8,3,1,  update_collection=update_collection, padStr="SAME"), name='bn4', is_training=train) ## 8x8 4 channels
		c4 = tf.nn.relu(c4)

		c5 = batch_norm(convLayer(c4, 5, 4,3,1, update_collection=update_collection, padStr="SAME"), name='bn5', is_training=train) ## 8x8 4 channels
		c5 = tf.nn.relu(c5)
		c5 = tf.reshape(c5, [N_BATCH, 8 * 8 *4])

		## dense layer
		d1 = batch_norm(denseLayer(c5, 4, 512, update_collection=update_collection), is_training=train, name="bn6")
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 512, spec_norm=True, update_collection=update_collection), is_training=train, name="bn7")
		d2 = tf.nn.relu(d2)
		#do2 = tf.layers.dropout(d2, rate=0.3, training=train)
		## Final conv layer
		d2 = tf.reshape(d2, [N_BATCH, 8,8, 8])
		c4 = batch_norm(convLayer(d2, 7, 8,3, 1, update_collection=update_collection,  padStr="SAME"), is_training=train, name="bn8")
		c4 = tf.reduce_mean(c4,3)		
		c4 = tf.nn.relu(c4) ## output activation

		## Reshape and output
		x = tf.reshape(c4, [N_BATCH, 8, 8]) ## make sure this is correct for addition with X_REAL in interpolate
		return x
		
""" --------------- Critic graph ---------------------------------------------------------"""	
def discriminator(x, y, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("discriminator", reuse=True) as scope:
		print("Preparing critic graph...")

		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = layer_norm(convLayer(y, 1, 4, 7, 3, spec_norm=False, update_collection=update_collection), is_training=train, name="ln1") ## 32x32, 4 chhannels
		c1 = tf.nn.relu(c1)
		c2 = layer_norm(convLayer(c1, 2, 8, 5, 3, spec_norm=False, update_collection=update_collection), is_training=train, name="ln2") ## 10x10 4 channels
		c2 = tf.nn.relu(c2) 
		c3 = layer_norm(convLayer(c2, 3, 8,3,1, spec_norm=False, update_collection=update_collection), is_training=train, name="ln3") ## 8x8 4 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8, 8, 8])
		
		## concat with real/fake fourier coefficients
		x = tf.reshape(x, [N_BATCH, 8,8,1]) # fourier space - 64
		concat = tf.concat([x,c], 3) # concat along second dimension

		## go through additional conv layers to enforce locality in feedback
		c4 = convLayer(concat, 4, 8,3,1, spec_norm=False, update_collection=update_collection, padStr="SAME") ## 8x8 4 channels
		c4 = tf.nn.leaky_relu(c4)

		c5 = convLayer(c4, 5, 8,3,1, spec_norm=False, update_collection=update_collection, padStr="SAME") ## 8x8 4 channels
		c5 = tf.nn.leaky_relu(c5)
		## no go through dense layers

		c5 = tf.reshape(c5, [N_BATCH, 8 * 8 * 8])

		## dense layer

		d1 = denseLayer(c5, 6, 512, spec_norm=False, update_collection=update_collection)
		d1 = tf.nn.leaky_relu(d1)
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)		

		d2 = denseLayer(do1, 7, 256, spec_norm=False, update_collection=update_collection)
		d2 = tf.nn.leaky_relu(d2)
		
		do2 = tf.layers.dropout(d2, rate=0.1, training=train)
		
		d3 = denseLayer(do2, 8, 1, spec_norm=False, update_collection=update_collection)
		## Reshape and output
		D = tf.reshape(d3, [N_BATCH, 1])
		return D ## wasserstein 

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
	path = "C:\\Jannes\\learnSamples\\040319_1W_0001s\\validation"
	outPath = "C:\\Jannes\\learnSamples\\040319_validation\\cWGAN_recLoss"
	
	## Check PATHS
	if not os.path.exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not os.path.exists(outPath):
    		os.makedirs(outPath)

	fourier_folder = "inFourier"
	input_folder = 	"in"
	output_folder = "out"
	minFileNr = 1
	indices = get_file_indices(os.path.join(path, output_folder))
	maxFile = len(indices) ## number of samples in data set

	#############################################################################
	restore = True ### Set this True to load model from disk instead of training
	testSet = False
	#############################################################################

	save_name = "W_GP_HOLOGAN.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta_D = 0.0001
	eta_G = 0.0001
	N_BATCH = 60
	N_VALID = 100
	N_REDRAW = 5	
	N_CRITIC = 5
	N_EPOCH = 20
	N_LAT = 64
	LAMBDA = 10
	BETA = 1.0
	## sample size
	N_SAMPLE = maxFile-N_BATCH*N_CRITIC
	last_index  = 0
	print("Data set has length "+str(N_SAMPLE))

	### Define file load functions
	load_fourier = lambda x, nr : 1.0/100*np.squeeze(load_files(os.path.join(path, fourier_folder), nr, minFileNr+ x, indices))
	load_input = lambda x, nr : 1.0/255*np.squeeze(load_files(os.path.join(path, input_folder), nr, minFileNr + x, indices))
	load_output = lambda x, nr: 1.0/255*np.squeeze(load_files(os.path.join(path, output_folder), nr, minFileNr + x, indices))

	""" --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	X_REAL = tf.placeholder(dtype=tf.float32, shape=(N_BATCH, 8,8), name="X_REAL") ## Fourier input
	Z = tf.placeholder(dtype=tf.float32, name="Z") ## Latent variables
	Y = tf.placeholder(dtype=tf.float32, name="Y")
		
	# ROUTE THE TENSORS
	X_FAKE = generator(Z,Y, is_train, N_LAT, N_BATCH) ## GENERATOR GRAPH
	D_REAL = discriminator(X_REAL, Y, is_train, N_BATCH, update_collection="NO_OPS") ## REAL CRITIC GRAPH
	D_FAKE = discriminator(X_FAKE,Y, is_train,N_BATCH, update_collection=None)

	# Loss functions	
	# WGAN loss
	alpha = tf.random_uniform(shape=[N_BATCH,1,1], minval=0., maxval=1.)
	D_loss = tf.reduce_mean(D_FAKE) - tf.reduce_mean(D_REAL)
	G_loss = -tf.reduce_mean(D_FAKE) + BETA/64*tf.nn.l2_loss(X_FAKE-X_REAL)

	## gradient penalty
	interpolates = alpha*X_REAL + ((1-alpha)*X_FAKE)
	D_interpolates = discriminator(interpolates, Y, is_train, N_BATCH, update_collection=None)
    	
	gradients = tf.gradients(D_interpolates, [interpolates])[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
	gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
	D_loss += LAMBDA*gradient_penalty

	# Group variables
	#tvars = tf.trainable_variables()
	D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")#[var for var in tvars if 'critic' in var.name]
	G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")#[var for var in tvars if 'generator' in var.name]	
	
	# Trainin operations
	D_solver = tf.train.AdamOptimizer(
            learning_rate=eta_D, 
            beta1=0.5, 
            beta2=0.9).minimize(D_loss, var_list=D_vars)

	G_solver = tf.train.AdamOptimizer(
            learning_rate=eta_G, 
            beta1=0.5, 
            beta2=0.9).minimize(G_loss, var_list=G_vars)

	
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
		
		D_loss_array = [] # Discriminator loss
		G_loss_array = [] # Generator loss
		percent = 0

		N_STEPS = N_EPOCH*N_SAMPLE/N_BATCH
		if not restore :
			for j in range(N_EPOCH):   	
				for i in range(0, N_SAMPLE, N_CRITIC*N_BATCH):
					## Train generator
					x = load_fourier(i, N_BATCH)
					y = load_output(i, N_BATCH)
					z = sample_Z(N_BATCH, N_LAT) 
					sess.run(G_solver, feed_dict={X_REAL: x, Y: y, Z: z, is_train: True})
					
					if i+(N_CRITIC-1)*N_BATCH < N_SAMPLE: ## make sure this is within the index bounds
						## Train critic
						for k in range(N_CRITIC):
							x = load_fourier(i+k*N_BATCH, N_BATCH)	
							y = load_output(i+k*N_BATCH, N_BATCH)
							z = sample_Z(N_BATCH, N_LAT)
							sess.run(D_solver, feed_dict={X_REAL: x, Y: y, Z: z, is_train: True})
							
					## store the progress
					if int(100 * ( (j*N_SAMPLE+i)/float(N_SAMPLE*N_EPOCH))) != percent :
						percent = int( 100 * ((j*N_SAMPLE+ i)/float(N_SAMPLE*N_EPOCH)))
					
						curr_D_loss = sess.run(D_loss, feed_dict={X_REAL: x, Y: y, Z: z, is_train: False})
						curr_G_loss = sess.run(G_loss, feed_dict={X_REAL: x, Y: y, Z: z, is_train: False})					
						D_loss_array.append(curr_D_loss)
						G_loss_array.append(curr_G_loss)
						print(str(percent) + "% ## D loss " + str(curr_D_loss) + " | G loss " + str(curr_G_loss))
				
					
			    
			plt.figure(figsize=(8, 8))
			plt.plot(np.array(D_loss_array), 'r-')
			plt.plot(np.array(G_loss_array), 'b-')
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
					z = sample_Z(N_BATCH, N_LAT)
					x_pred = sess.run(X_FAKE, feed_dict={Y:y, Z:z, is_train: False})

					## write the matrices to file
					if testSet:
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
					else:				
						writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))

			print("DONE! :)")
if __name__ == "__main__":
	main(sys.argv)
