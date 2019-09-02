from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import date
from libs.input_helper import *
from libs.ops import *

""" --------------- FORWARD Graph -------------------------------------------------------------"""
def forward(x, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("forward", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up forward graph")
		x = tf.reshape(x, [N_BATCH, 8,8,2]) ## shoule be in this shape anyway
		c1 = convLayer(x, 1, 4,3,1, spec_norm=True,  update_collection=update_collection, padStr="SAME")## 8x8 4 channels
		c1 = tf.nn.relu(c1)
		c2 = convLayer(c1, 2, 8,3,1,  spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
		c2 = tf.nn.relu(c2)
		c3 = convLayer(c2, 3, 16,5,1,  spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 16 channels
		c3 = tf.nn.relu(c3)
		
		# dropout
		do = tf.layers.dropout(c3, rate=0.2, training=train)
		# a single dense layer to account for nonlocal effects
		c4 = convLayer(do, 4, 16, 5,1,  spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 16 channels
		c4 = tf.nn.relu(c4)

		## Deconvolution Layers
		dc0 =  deconvLayer(c4, 5, [N_BATCH, 10,10, 16], 3, 1, spec_norm=True, update_collection=update_collection )# 10x10, 16 channels
		dc0 = tf.nn.relu(dc0)
		dc1 =  deconvLayer(dc0, 6, [N_BATCH, 32,32,8], 5, 3, spec_norm=True, update_collection=update_collection )# 32x32, 8 channels
		dc1 = tf.nn.relu(dc1)
		dc2 =  deconvLayer(dc1, 7, [N_BATCH, 100,100,8], 7, 3, spec_norm=True, update_collection=update_collection)# 100x100, 8 channels
		dc  = tf.reduce_mean(dc2, 3) ## collapse channels 		
		
		y = tf.nn.relu(dc) ## [-1, 100, 100]

		return y
""" --------------- DECODER Graph --------------------------------------------------------------"""		
def decoder(z, y, train, N_LAT, N_BATCH,  update_collection=tf.GraphKeys.UPDATE_OPS): ## output an x estimate
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up the decoder graph...")
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 12, 7, 3,  update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 8 chhannels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 16, 5, 3,  update_collection=update_collection), name='bn2', is_training=train) ## 10x10 8 channels
		c2 = tf.nn.relu(c2) 
		do0 = tf.layers.dropout(c2, rate=0.2, training=train)		
		c3 = batch_norm(convLayer(do0, 3, 16, 3, 1,  update_collection=update_collection), name='bn3', is_training=train) ## 8x8 8 channels
		c3 = tf.nn.relu(c3) 
		## Now combine with the latent variables
		z = tf.tile(z, [8,8])
		z = tf.reshape(z, [N_BATCH, 8,8,1]) # latent space, linear
		c = tf.reshape(c3, [N_BATCH, 8,8,16]) ## 8x8x8 = 512
		concat = tf.concat([z,c], 3) # 512 + 16 = 528
		
		## dense layer -- probably required since not everything is local 
		c4 = batch_norm(convLayer(concat, 4, 16, 3, 1,  update_collection=update_collection, padStr="SAME"), name='bn4', is_training=train) ## 8x8 8 channels
		c4 = tf.nn.relu(c4)
		c5 = batch_norm(convLayer(c4, 5, 16, 3, 1,  update_collection=update_collection, padStr="SAME"), name='bn5', is_training=train) ## 8x8 8 channels
		c5 = tf.nn.relu(c4)
		c6 = batch_norm(convLayer(c5, 6, 16, 3, 1,  update_collection=update_collection, padStr="SAME"), name='bn6', is_training=train) ## 8x8 8 channels
		c6 = tf.nn.relu(c6)
		do1 = tf.layers.dropout(c6, rate=0.2, training=train)		

		## Final conv layers
		## split in absolute and phase parts
		d2_abs = do1[:,:,:,0:8]
		d2_phi = do1[:,:,:,8:16]
		## go through final conv layer(s) each to use/enforce locality
		cf_abs = batch_norm(convLayer(d2_abs, 7, 8, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn7', is_training=train) ## 8x8 4 channel
		cf_phi = batch_norm(convLayer(d2_phi, 8, 8, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn8', is_training=train) ## 8x8 4 channel
		cf_abs = tf.nn.relu(cf_abs)
		cf_phi = tf.nn.relu(cf_phi)
		cf_abs = batch_norm(convLayer(cf_abs, 9, 8, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn9', is_training=train) ## 8x8 1 channel
		cf_phi = batch_norm(convLayer(cf_phi, 10, 8, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn10', is_training=train) ## 8x8 1 channel
		# collapse channel dimension
		cf_abs = tf.nn.relu( tf.reduce_mean(cf_abs, axis=3) ) ## absolute values, retain channel dimension...
		cf_phi = tf.nn.relu( tf.reduce_mean(cf_phi, axis=3) ) ## angles, retain channel dimension...
		
		## final reshaping and return prediction
		x_hat = tf.concat([cf_abs[:,:,:,None], cf_phi[:,:,:,None]], axis=3) ## ... in order to be able to concat along it.
		return x_hat
""" --------------- ENCODER Graph --------------------------------------------------------------"""		
def encoder(x,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS): ## output some gaussian parameters
	with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up encoder graph...")
		
		### y dependence of encoder not neeeded - y noise is small, x contains all information
		"""y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 8 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 8 channels
		c3 = tf.nn.relu(c3)
		do0 = tf.layers.dropout(c3, rate=0.2, training=train)
		c = tf.reshape(do0, [N_BATCH, 8 * 8 * 8])"""

		## combine reduced conditional variable with setup input - x
		x = tf.reshape(x, [N_BATCH, 2*8*8]) # fourier space - 64
		#concat = tf.concat([x,c], 1) # concat along channel dimension
		## dense layer
		d1 = batch_norm(denseLayer(x, 4, 128, update_collection=update_collection), name='bn4', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.2, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 64, update_collection=update_collection), name='bn5', is_training=train)
		d2 = tf.nn.relu(d2)

		d3 = batch_norm(denseLayer(d2, 6, 2*N_LAT, update_collection=update_collection), name='bn6', is_training=train)
		## Reshape and output		
		lat_par = tf.reshape(d3, [N_BATCH, N_LAT, 2]) ## [:, :, 0] -> means, [:,:,1] -> log_sigma
		return lat_par
""" --------------------------------------------------------------------------------------------"""		
def unload_gauss_args(lat):
	mu = lat[:,:,0] ## [N_BATCH, N_LAT] #tf.squeeze(tf.slice(lat, [0, 0, 0], [-1, -1, 1]))
	log_sigma_sq = lat[:,:,1] ## [N_BATCH, N_LAT] #tf.squeeze(tf.slice(lat, [0, 0, 1], [-1, -1, 1]))
	return mu, log_sigma_sq

# sampling & reparametrization 
def sample(lat, N_LAT, N_BATCH): 
	mu, log_sigma_sq = unload_gauss_args(lat) # both are [N_BATCH, 64]
	eps = tf.random.normal([N_BATCH, N_LAT], mean=0.0, stddev=1.)

	return tf.add(mu, tf.multiply(tf.exp(log_sigma_sq/2), eps)) ## [N_BATCH, 64]

def setup_vae_loss(x, x_hat, lat, BETA, N_SAMPLE, N_EPOCH, N_BATCH):
	""" Calculate loss = reconstruction loss + KL loss for each data in minibatch """
	mu, log_sigma = unload_gauss_args(lat)
	# <log P(x | z, y)>	

	reconstruction_loss = BETA*tf.nn.l2_loss(x-x_hat) #tf.reduce_mean(tf.losses.absolute_difference(x, x_hat)) 
	# KL(Q(z | x, y) || P(z | x)) - in analytical form
	kullback_leibler =  0.5 * tf.reduce_sum(tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma) ## -KL term

	return reconstruction_loss + kullback_leibler
""" ----------- MAIN ---------------------------------------------------------------------------"""		
def main(argv):

	#############################################################################
	path = r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw\validation"
	outPath = r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw\models\cVAE_model_forward_2"
	restore = True ### Set this True to load model from disk instead of training
	testSet = True
	#############################################################################

	## Check PATHS
	if not os.path.exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not os.path.exists(outPath):
		print("MODEL/OUTPUT PATH DOESN'T EXIST!")
		sys.exit()
	
	### Define file load functions
	data = data_obj(path, shuffle_data= not (restore or testSet), test_set=testSet )

	save_name = "HOLOVAE_FORWARD.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta = 1e-4
	eta_f = 1e-4
	N_BATCH = 100
	N_VALID = 500	
	N_REDRAW = 5	
	N_EPOCH = 15
	N_LAT = 1
	BETA = 1.0
	ALPHA = 1.0
	## sample size
	N_SAMPLE = data.maxFile-N_BATCH
	print("Data set has length {}".format(N_SAMPLE))

	""" --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	Y = tf.placeholder(shape=(N_BATCH, 100,100), dtype=tf.float32, name="Y")
	X = tf.placeholder(shape=(N_BATCH, 8,8,2), dtype=tf.float32, name="X")
	# ROUTE THE TENSORS
	LAT = encoder(X, Y, is_train, N_LAT, N_BATCH)
	Z = sample(LAT, N_LAT, N_BATCH)
	X_HAT = decoder(Z,Y, is_train, N_LAT, N_BATCH) ## GENERATOR GRAPH [N_BATCH, 8,16]
	Y_HAT = forward(X, is_train, N_BATCH)
	Y_HAT_HAT = forward(X_HAT, is_train, N_BATCH)
	## VALIDATION TENSORS
	Z_VALID = tf.random.normal([N_BATCH, N_LAT], mean=0.0, stddev=1)
	X_HAT_VALID = decoder(Z_VALID, Y, is_train, N_LAT, N_BATCH)

	VAE_var_list = list(set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="decoder")).union( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")))
	FORW_var_list =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="forward")

	## Loss functions	
	VAE_loss = setup_vae_loss(X, X_HAT, LAT, BETA, N_SAMPLE, N_EPOCH, N_BATCH)
	Y_loss = tf.nn.l2_loss(Y - Y_HAT)
	Y_HAT_loss = tf.nn.l2_loss(Y_HAT_HAT-Y)
	X_loss = tf.nn.l2_loss(X - X_HAT)
	VAE_solver = tf.train.RMSPropOptimizer(learning_rate=eta).minimize(VAE_loss + ALPHA*Y_HAT_loss, var_list=VAE_var_list)
	FORW_solver = tf.train.AdamOptimizer(learning_rate=eta_f).minimize(Y_loss, var_list=FORW_var_list)
	# Initializer
	initializer = tf.global_variables_initializer() # get initializer   
	
	if testSet:
    		restore = True
	""" -------------- TRAINING ---------------------------------------------------------------------"""
	with tf.Session() as sess:
		with save_on_exit(sess, save_string) as save_guard:

			sess.run(initializer)    
			if not restore :
				x_err = []
				y_hat_err = []
				vae_err = []
				percent=0
				### forward network pretraining
				print("Commencing pretraining...")

				for i in range(0, N_SAMPLE, N_BATCH):
						x = data.load_fourier(i, N_BATCH)
						y = data.load_output(i, N_BATCH)
						sess.run(FORW_solver, feed_dict={X:x, Y:y, is_train:True})

				print("Commencing training...")

				### main training		
				for j in range(N_EPOCH):   
					for i in range(0, N_SAMPLE, N_BATCH):
						if int(100 * ( (j*N_SAMPLE+i)/float(N_SAMPLE*N_EPOCH))) != percent :
							percent = int( 100 * ((j*N_SAMPLE+ i)/float(N_SAMPLE*N_EPOCH)))
							x = data.load_fourier(i, N_BATCH)
							y = data.load_output(i, N_BATCH)
							vae_loss = sess.run(VAE_loss, feed_dict={X:x, Y:y, is_train:False} )
							y_hat_loss = sess.run(Y_HAT_loss, feed_dict={X:x, Y:y, is_train:False})
							x_loss = sess.run(X_loss, feed_dict={X:x, Y:y, is_train: False})
							x_err.append(x_loss)
							y_hat_err.append(y_hat_loss)
							vae_err.append(vae_loss)
							print(str( percent ) + "%"+ " ## xloss " + str(x_loss) + "  ## yloss "+ str(y_hat_loss) +" ## VAE loss " + str(vae_loss))

						x = data.load_fourier(i, N_BATCH)
						y = data.load_output(i, N_BATCH)
						sess.run(FORW_solver, feed_dict={X:x, Y:y, is_train:True})
						sess.run(VAE_solver, feed_dict={X:x, Y:y, is_train: True})			
			
				plt.figure(figsize=(8, 8))
				plt.plot(np.array(x_err), 'r-')
				plt.plot(np.array(vae_err), 'b-')
				plt.plot(np.array(y_hat_err), 'k-')
				plt.show()

				#### Return and save #########		
				return 

			#### RESTORE MODEL& apply to validate #####
			elif restore:
				save_guard.restore_model()

				#### VALIDATION ########
				for k in range(0, N_VALID):
					testNr =  k
					if not testSet:
						x = data.load_fourier(testNr, N_BATCH)
					y = data.load_output(testNr, N_BATCH)
					for r in range(N_REDRAW):
						fileNr =  k*N_REDRAW + r
						## draw new noise
						x_pred = sess.run(X_HAT_VALID, feed_dict={Y:y, is_train: False})
						## write the matrices to file
						if testSet:
							writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
						else:
							#plotMatrices(np.squeeze(x_pred[0,:,:,0]), np.squeeze(x[0,:,:,0]), np.squeeze(x_pred[0,:,:,1]), np.squeeze(x[0,:,:,1]))					
							writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))
							#y_pred = sess.run(Y_HAT, feed_dict={X:x, is_train:False})
							#plot_forward(y[0], y_pred[0])
	print("DONE! :)")
if __name__ == "__main__":
	main(sys.argv)