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
		c1 = batch_norm(convLayer(x, 1, 8,3,1, spec_norm=False,  update_collection=update_collection, padStr="SAME"), name='bn1', is_training=train) ## 8x8 8 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8,3,1,  spec_norm=False, update_collection=update_collection, padStr="SAME"), name='bn2', is_training=train) ## 8x8 8 channels
		c2 = tf.nn.relu(c2)
        ## Dense Layer
		c = tf.reshape(c2, [N_BATCH, 8*8*8])

		# dropout
		do1 = tf.layers.dropout(c, rate=0.2, training=train)
		# a single dense layer to account for nonlocal effects
		d1 = batch_norm(denseLayer(do1, 4, 400, spec_norm=False, update_collection=update_collection), name='bn3', is_training=train)
		d1 = tf.nn.relu(d1)

		d = tf.reshape(d1, [N_BATCH, 10,10,4])
		
		## Deconvolution Layers
		dc1 =  batch_norm(deconvLayer(d, 6, [N_BATCH, 32,32,4], 5, 3, spec_norm=False, update_collection=update_collection ), name='bn4', is_training=train) # 32x32, 4 channels
		dc1 = tf.nn.relu(dc1)
		dc2 =  batch_norm(deconvLayer(dc1, 7, [N_BATCH, 100,100,4], 7, 3, spec_norm=False, update_collection=update_collection), name='bn5', is_training=train) # 100x100, 4 channels
		dc  = tf.reduce_mean(dc2, 3) ## collapse channels 		
		
		y = tf.nn.relu(dc) ## [-1, 100, 100]

		return y
""" --------------- FORWARD Graph -------------------------------------------------------------"""
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

		## dropout to ensure that filters catch with redundancy
		do = tf.layers.dropout(c3, rate=0.2, training=train)
		c = tf.reshape(do, [N_BATCH, 8*8*8])

		## Now combine with the latent variables
		z = tf.reshape(z, [N_BATCH,N_LAT]) # latent space - 64		
		
		concat = tf.concat([z,c], 1) # concat along thirddimension
		d0 = batch_norm(denseLayer(concat, 4, 512, update_collection=update_collection), name='bn4', is_training=train)
		d0 = tf.nn.relu(d0)
		## dropout from dense layers
		do0 = tf.layers.dropout(d0, rate=0.2, training=train)

		## dense layer -- probably required since not everything is local 
		d1 = batch_norm(denseLayer(do0, 5, 512, update_collection=update_collection), name='bn5', is_training=train)
		d1 = tf.nn.relu(d1)
		## dropout from dense layers
		do1 = tf.layers.dropout(d1, rate=0.2, training=train)
		
		d2 = batch_norm(denseLayer(do1, 6, 512, update_collection=update_collection), name='bn6', is_training=train)
		d2 = tf.nn.relu(d2)
		
		## Final conv layers
		d2 = tf.reshape(d2, [N_BATCH, 8, 8, 8])
		#c6 = batch_norm(convLayer(d2, 7, 8,3, 1, update_collection=update_collection,  padStr="SAME"), name='bn8', is_training=train)
		#cf = batch_norm(convLayer(c6, 7, 8,3, 1, update_collection=update_collection,  padStr="SAME"), name='bn9', is_training=train)
		## split in absolute and phase parts
		cf_abs = tf.nn.relu( tf.reduce_mean(d2[:,:,:,0:4], axis=3)) ## absolute values
		cf_phi = tf.nn.relu( tf.reduce_mean(d2[:,:,:,4:8], axis=3)) ## angles
		
		#cf_abs = tf.nn.relu( tf.reduce_mean(cf[:,:,:,0:4], axis=3)) ## absolute values
		#cf_phi = tf.nn.relu( tf.reduce_mean(cf[:,:,:,4:8], axis=3)) ## angles
		
		## final reshaping and return prediction
		x_hat = tf.concat([cf_abs[:,:,:,None], cf_phi[:,:,:,None]], axis=3)
		return x_hat

""" --------------- ENCODER Graph --------------------------------------------------------------"""		

def encoder(x,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS): ## output some gaussian parameters
	with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up encoder graph...")
		#y = tf.reshape(y, [N_BATCH, 100, 100,1])
		#c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		#c1 = tf.nn.relu(c1)
		#c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 8 channels
		#c2 = tf.nn.relu(c2) 
		#c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 8 channels
		#c3 = tf.nn.relu(c3)
		#do0 = tf.layers.dropout(c3, rate=0.2, training=train)

		#c = tf.reshape(do0, [N_BATCH, 8 * 8 * 8])
		## combine reduced conditional variable with setup input - x
		x = tf.reshape(x, [N_BATCH, 8,8,2]) # fourier space - 64
		#concat = tf.concat([x,c], 1) # concat along channel dimension
		## dense layer
		c =  batch_norm(convLayer(x, 1, 4,3,1, update_collection=update_collection, padStr="SAME"), name='bn0', is_training=train) 
		c = tf.reshape(c, [N_BATCH, 8*8*4]) # fourier space - 64
		
		d1 = batch_norm(denseLayer(c, 4, 512, update_collection=update_collection), name='bn1', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.2, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 256, update_collection=update_collection), name='bn2', is_training=train)
		d2 = tf.nn.relu(d2)
				
		d3 = batch_norm(denseLayer(d2, 6, 2*N_LAT, update_collection=update_collection), name='bn3', is_training=train)

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
	path = "C:\\Jannes\\learnSamples\\190719_blazedGrating_phase_redraw\\"
	outPath = "C:\\Jannes\\learnSamples\\190719_blazedGrating_phase_redraw\\models\\cVAE_model_2_FORWARD"
	restore = False ### Set this True to load model from disk instead of training
	testSet = False
	#############################################################################
	## Check PATHS
	if not os.path.exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not os.path.exists(outPath):
		print("MODEL/OUTPUT PATH DOESN'T EXIST!")
		sys.exit()
	
	### Define file load functions
	data = data_obj(path, shuffle_data= not (restore or testSet) )

	save_name = "HOLOVAE_FORWARD.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta = 1e-4
	eta_f = 1e-4
	N_BATCH = 100
	N_VALID = 100	
	N_REDRAW = 5	
	N_EPOCH = 15
	N_LAT = 16
	BETA = 1.0
	ALPHA = 5.0
	## sample size
	N_SAMPLE = data.maxFile-N_BATCH
	print("Data set has length {}".format(N_SAMPLE))

	""" --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	Y = tf.placeholder(dtype=tf.float32, name="Y")
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
	VAE_solver = tf.train.AdamOptimizer(learning_rate=eta).minimize(VAE_loss + ALPHA*Y_HAT_loss, var_list=VAE_var_list)
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

	print("DONE! :)")
if __name__ == "__main__":
	main(sys.argv)