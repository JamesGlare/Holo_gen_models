from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import date
from libs.ops import *
from libs.input_helper import *
""" --------------- FORWARD Graph -------------------------------------------------------------"""
def forward(x, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("forward", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up forward graph")

		x = tf.reshape(x, [N_BATCH, 8,8,2]) ## shoule be in this shape anyway
		c1 = convLayer(x, 1, 8,3,1, spec_norm=True,  update_collection=update_collection, padStr="SAME")## 8x8 8 channels
		c1 = tf.nn.relu(c1)
		c2 = convLayer(c1, 2, 8,3,1,  spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
		c2 = tf.nn.relu(c2)
        ## Dense Layer
		c = tf.reshape(c2, [N_BATCH, 8*8*8])

		# dropout
		do1 = tf.layers.dropout(c, rate=0.2, training=train)
		# a single dense layer to account for nonlocal effects
		d1 = denseLayer(do1, 4, 400, spec_norm=True, update_collection=update_collection)
		d1 = tf.nn.relu(d1)

		d = tf.reshape(d1, [N_BATCH, 10,10,4])
		
		## Deconvolution Layers
		dc1 =  deconvLayer(d, 6, [N_BATCH, 32,32,4], 5, 3, spec_norm=True, update_collection=update_collection )# 32x32, 4 channels
		dc1 = tf.nn.relu(dc1)
		dc2 =  deconvLayer(dc1, 7, [N_BATCH, 100,100,4], 7, 3, spec_norm=True, update_collection=update_collection)# 100x100, 4 channels
		dc  = tf.reduce_mean(dc2, 3) ## collapse channels 		
		
		y = tf.nn.relu(dc) ## [-1, 100, 100]

		return y
""" --------------- Generator Graph ----------------------------------------------------------"""		
def generator(z,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("generator") as scope:
		print("Setting up the generator graph...")
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 8, 7, 3,  update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 8 chhannels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3,  update_collection=update_collection), name='bn2', is_training=train) ## 10x10 8 channels
		c2 = tf.nn.relu(c2) 
		do0 = tf.layers.dropout(c2, rate=0.2, training=train)		
		c3 = batch_norm(convLayer(do0, 3, 8, 3, 1,  update_collection=update_collection), name='bn3', is_training=train) ## 8x8 8 channels
		c3 = tf.nn.relu(c3) 
		## Now combine with the latent variables
		z = tf.reshape(z, [N_BATCH, N_LAT]) # latent space, linear
		c = tf.reshape(c3, [N_BATCH, 512]) ## 8x8x8 = 512
		concat = tf.concat([z,c], 1) # 512 + 16 = 528
		
		## dense layer -- probably required since not everything is local 
		d1 = batch_norm(denseLayer(concat, 4, 512, update_collection=update_collection), name='bn4', is_training=train)
		d1 = tf.nn.relu(d1)
		## dropout from dense layers
		do1 = tf.layers.dropout(d1, rate=0.2, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 256, update_collection=update_collection), name='bn5', is_training=train)
		d2 = tf.nn.relu(d2)

		d3 = batch_norm(denseLayer(d2, 6, 200, update_collection=update_collection), name='bn6', is_training=train)
		d3 = tf.nn.relu(d3)
		
		## Final conv layers
		d = tf.reshape(d3, [N_BATCH, 10, 10, 2])
		## split in absolute and phase parts
		d2_abs = d[:,:,:,0]
		d2_phi = d[:,:,:,1]
		## go through final conv layer(s) each to use/enforce locality
		cf_abs = batch_norm(convLayer(d2_abs[:,:,:,None], 7, 4, 3, 1, update_collection=update_collection), name='bn7', is_training=train) ## 8x8 4 channel
		cf_phi = batch_norm(convLayer(d2_phi[:,:,:,None], 8, 4, 3, 1, update_collection=update_collection), name='bn8', is_training=train) ## 8x8 4 channel
		cf_abs = tf.nn.relu(cf_abs)
		cf_phi = tf.nn.relu(cf_phi)
		cf_abs = batch_norm(convLayer(cf_abs, 9, 1, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn9', is_training=train) ## 8x8 1 channel
		cf_phi = batch_norm(convLayer(cf_phi, 10, 1, 3, 1, update_collection=update_collection, padStr="SAME"), name='bn10', is_training=train) ## 8x8 1 channel
		# collapse channel dimension
		cf_abs = tf.nn.relu( cf_abs ) ## absolute values, retain channel dimension...
		cf_phi = tf.nn.relu( cf_abs)  ## angles, retain channel dimension...
		
		## final reshaping and return prediction
		x_hat = tf.concat([cf_abs, cf_phi], axis=3) ## ... in order to be able to concat along it.
		return x_hat
""" --------------- Critic graph ---------------------------------------------------------"""	
def discriminator(x, y, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("discriminator", reuse=True) as scope:
		print("Preparing critic graph...")

		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = convLayer(y, 1, 4, 7, 3, spec_norm=True, update_collection=update_collection) ## 32x32, 4 chhannels
		c1 = tf.nn.relu(c1)
		c2 = convLayer(c1, 2, 8, 5, 3, spec_norm=True, update_collection=update_collection) ## 10x10 8 channels
		c2 = tf.nn.relu(c2) 
		do0 = tf.layers.dropout(c2, rate=0.2, training=train)		

		c3 = convLayer(do0, 3, 8, 3, 1, spec_norm=True, update_collection=update_collection) ## 8x8 8 channels
		c = tf.nn.relu(c3)
		
		## concat with real/fake fourier coefficients
		x = tf.reshape(x, [N_BATCH, 8,8,2]) # fourier space - 64
		concat = tf.concat([x,c], 3) # concat along second dimension

		## go through additional conv layers to enforce locality in feedback
		c4 = convLayer(concat, 4, 8, 3, 1, spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 4 channels
		c4 = tf.nn.leaky_relu(c4)
		## no go through dense layers

		cf = tf.reshape(c4, [N_BATCH, 8 * 8 * 8])

		d1 = denseLayer(cf, 5, 512, spec_norm=True, update_collection=update_collection)
		d1 = tf.nn.leaky_relu(d1)
		do1 = tf.layers.dropout(d1, rate=0.2, training=train)		

		d2 = denseLayer(do1, 6, 256, spec_norm=True, update_collection=update_collection)
		d2 = tf.nn.leaky_relu(d2)
				
		D = denseLayer(d2, 7, 1, spec_norm=True, update_collection=update_collection)
		return D ## Disc assessment 
""" --------------- GAN Lib ------------------------------------------------------------"""	
def sample_Z(N_BATCH, N_LAT):
    return np.random.uniform(0, 1, size=[N_BATCH, N_LAT])
""" --------------- Main function ------------------------------------------------------------"""	
def main(argv):
	#############################################################################
	path = r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw"
	outPath = r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw\models\cGAN_forward"
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
	data = data_obj(path, shuffle_data= not (restore or testSet), test_set=testSet )

	save_name = "HOLOGAN.ckpt"
	save_string = os.path.join(outPath, save_name)

	### Hyperparameters
	tf.set_random_seed(42)
	eta_D = 0.0001
	eta_G = 0.0001
	N_BATCH = 100
	N_VALID = 500	
	N_CRITIC = 5
	N_REDRAW = 5	
	N_EPOCH = 20
	N_LAT = 16
	BETA = 1.0
	ALPHA = 1.0
	## sample size
	N_SAMPLE =  data.maxFile-N_BATCH
	print("Data set has length {}".format(N_SAMPLE))

	""" --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
	is_train = tf.placeholder(dtype=tf.bool, name="is_train")
	X_REAL = tf.placeholder(dtype=tf.float32, name="X_REAL") ## Fourier input
	Z = tf.placeholder(dtype=tf.float32, name="Z") ## Latent variables
	Y = tf.placeholder(dtype=tf.float32, name="Y")
		
	# ROUTE THE TENSORS
	X_FAKE = generator(Z,Y, is_train, N_LAT, N_BATCH) ## GENERATOR GRAPH
	D_REAL = discriminator(X_REAL, Y, is_train, N_BATCH, update_collection="NO_OPS") ## REAL CRITIC GRAPH
	D_FAKE = discriminator(X_FAKE,Y, is_train,N_BATCH, update_collection=None)
	Y_HAT = forward(X, is_train, N_BATCH)
	Y_HAT_HAT = forward(X_FAKE, is_train, N_BATCH)
	
	# Loss functions	
	Y_loss = tf.nn.l2_loss(Y - Y_HAT)
	Y_HAT_loss = tf.nn.l2_loss(Y_HAT_HAT-Y)
	D_loss = tf.reduce_mean(tf.nn.softplus(D_FAKE) + tf.nn.softplus(-D_REAL))	
	G_loss = tf.reduce_mean(tf.nn.softplus(-D_FAKE)) + BETA*tf.nn.l2_loss(X_FAKE-X_REAL)+ ALPHA*Y_HAT_loss

	
	# Group variables
	#tvars = tf.trainable_variables()
	D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")#[var for var in tvars if 'critic' in var.name]
	G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")#[var for var in tvars if 'generator' in var.name]	
	F_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="forward")
	
	# Trainin operations
	D_solver = tf.train.AdamOptimizer(
            learning_rate=eta_D, 
            beta1=0.5, 
            beta2=0.9).minimize(D_loss, var_list=D_vars)

	G_solver = tf.train.AdamOptimizer(
            learning_rate=eta_G, 
            beta1=0.5, 
            beta2=0.9).minimize(G_loss, var_list=G_vars)
	FORW_solver = tf.train.AdamOptimizer(learning_rate=eta_f).minimize(Y_loss, var_list=FORW_var_list)

	# Initializer
	initializer = tf.global_variables_initializer() # get initializer   

	if testSet:
		restore = True
	""" -------------- TRAINING ---------------------------------------------------------------------"""
	with tf.Session() as sess:
		with save_on_exit(sess, save_string) as save_guard:

			sess.run(initializer)
			
			D_loss_array = [] # Discriminator loss
			G_loss_array = [] # Generator loss
			percent = 0
			if not restore :
    			print("Commencing pretraining...")

				for i in range(0, N_SAMPLE, N_BATCH):
						x = data.load_fourier(i, N_BATCH)
						y = data.load_output(i, N_BATCH)
						sess.run(FORW_solver, feed_dict={X:x, Y:y, is_train:True})

				print("Commencing training...")
				for j in range(N_EPOCH):   	
					for i in range(0, N_SAMPLE, N_CRITIC*N_BATCH):
						## Train generator
						x = data.load_fourier(i, N_BATCH)
						y = data.load_output(i, N_BATCH)
						z = sample_Z(N_BATCH, N_LAT) 
						sess.run(G_solver, feed_dict={X_REAL: x, Y: y, Z: z, is_train: True})
						
						if i+(N_CRITIC-1)*N_BATCH < N_SAMPLE: ## make sure this is within the index bounds
							## Train critic
							for k in range(N_CRITIC):
								x = data.load_fourier(i+k*N_BATCH, N_BATCH)	
								y = data.load_output(i+k*N_BATCH, N_BATCH)
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
				#### Return and save #########		
				return

			#### RESTORE MODEL #####
			elif restore:
				save_guard.restore_model()
				
				for k in range(0, N_VALID):
					testNr = k
					if not testSet:
						x = data.load_fourier(testNr, N_BATCH)
					y = data.load_output(testNr, N_BATCH)
					for r in range(N_REDRAW):
						fileNr = k*N_REDRAW + r
						## draw new noise	
						z = sample_Z(N_BATCH, N_LAT)
						x_pred = sess.run(X_FAKE, feed_dict={ Y:y, Z:z, is_train: False}) 

						## write the matrices to file
						if testSet:
							writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
						else:
							writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))

	print("DONE! :)")


if __name__ == "__main__":
	main(sys.argv)
