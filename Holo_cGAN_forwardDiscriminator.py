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

def deconvLayer(x, nr, output_shape, kxy, stride, spec_norm=False, update_collection=tf.GraphKeys.UPDATE_OPS, padStr="VALID"):
		with tf.variable_scope("deconvLayer_"+str(nr), reuse=tf.AUTO_REUSE) as scope:
			return deconv2d(x, output_shape, k_h=kxy, k_w=kxy, name=str(nr), d_h=stride, d_w=stride, stddev=0.02, spectral_normed=spec_norm, update_collection=update_collection, padding=padStr)  

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

""" --------------- FORWARD Graph -------------------------------------------------------------"""
def forward(x, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("forward", reuse=tf.AUTO_REUSE) as scope:
		print("Setting up forward graph")

		x = tf.reshape(x, [N_BATCH, 8,8,1])
		c1 = convLayer(x, 1, 8,3,1, spec_norm=True,  update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
		c1 = tf.nn.relu(c1)
		c2 = convLayer(c1, 2, 8,3,1,  spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
		c2 = tf.nn.relu(c2)
        ## Dense Layer
		c = tf.reshape(c2, [N_BATCH, 8*8*8])

		d1 = denseLayer(c, 3, 512, spec_norm=True, update_collection=update_collection)
		d1 = tf.nn.relu(d1)
		# dropout
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		#
		d2 = denseLayer(do1, 4, 10000, specnorm=True, update_collection=update_collection)
		d2 = tf.nn.relu(d2)


		y = tf.reshape(d2, [N_BATCH, 100,100])
		
		## Deconvolution Layers
		#dc1 =  deconvLayer(d, 5, [N_BATCH, 10,10,4], 3, 1, spec_norm=True, update_collection=update_collection) # 10x10, 4 channels
		#dc1= tf.nn.relu(dc1)
		#dc2 =  deconvLayer(dc1, 6, [N_BATCH, 32,32,4], 5, 3, spec_norm=True, update_collection=update_collection ) # 32x32, 4 channels
		#dc2= tf.nn.relu(dc2)
		#dc3 =  deconvLayer(dc2, 7, [N_BATCH, 100,100,4], 7, 3, spec_norm=True, update_collection=update_collection) # 100x100, 4 channels
		#dc = tf.reduce_mean(dc3, 3) ## collapse channels 		
		
		#y = tf.nn.relu(dc) ## [-1, 100, 100]

		return y
""" --------------- Generator Graph ----------------------------------------------------------"""		
def generator(z,y, train, N_LAT, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
	with tf.variable_scope("generator") as scope:
		print("Preparing generator graph...")
		## LAYER 1 - conv for y
		y = tf.reshape(y, [N_BATCH, 100, 100,1])
		c1 = batch_norm(convLayer(y, 1, 4, 7, 3, update_collection=update_collection), name='bn1', is_training=train) ## 32x32, 4 channels
		c1 = tf.nn.relu(c1)
		c2 = batch_norm(convLayer(c1, 2, 8, 5, 3, update_collection=update_collection), name='bn2', is_training=train) ## 10x10, 4 channels
		c2 = tf.nn.relu(c2) 
		c3 = batch_norm(convLayer(c2, 3, 8,3,1, update_collection=update_collection), name='bn3', is_training=train) ## 8x8, 4 channels
		c3 = tf.nn.relu(c3)
		c = tf.reshape(c3, [N_BATCH, 8,8,8])
		## Now combine with the latent variables
		z =tf.reshape(z, [N_BATCH,8,8,1]) # latent space - 64
		concat = tf.concat([z,c], 3) # concat along second dimension
		
		## go through additional conv layers to enforce locality in feedback
		c4 = batch_norm(convLayer(concat, 4, 8,3,1,  update_collection=update_collection, padStr="SAME"), name='bn4', is_training=train) ## 8x8 4 channels
		c4 = tf.nn.relu(c4)

		c5 = batch_norm(convLayer(c4, 5, 4,3,1, update_collection=update_collection, padStr="SAME"), name='bn5', is_training=train) ## 8x8 4 channels
		c5 = tf.nn.relu(c5)
		c5 = tf.reshape(c5, [N_BATCH, 8 * 8 *4])

		## dense layer

		d1 = batch_norm(denseLayer(c5, 4, 512, update_collection=update_collection), name='bn6', is_training=train)
		d1 = tf.nn.relu(d1)
				
		do1 = tf.layers.dropout(d1, rate=0.3, training=train)
		
		d2 = batch_norm(denseLayer(do1, 5, 256, update_collection=update_collection), name='bn7', is_training=train)
		d2 = tf.nn.relu(d2)
		#do2 = tf.layers.dropout(d2, rate=0.3, training=train)
		## Final conv layer
		d2 = tf.reshape(d2, [N_BATCH, 8,8, 4])
		c4 = batch_norm(convLayer(d2, 7, 4, 3, 1, update_collection=update_collection,  padStr="SAME"), name='bn8', is_training=train)
		c4 = tf.reduce_mean(c4,3)				
		c4 = tf.nn.relu(c4) ## output activation

		## Reshape and output
		x = tf.reshape(c4, [N_BATCH, 8, 8]) ## make sure this is correct for addition with X_REAL in interpolate
		return x
		
""" --------------- Critic graph ---------------------------------------------------------"""	
def discriminator(y, train, N_BATCH, update_collection=tf.GraphKeys.UPDATE_OPS):
    with tf.variable_scope("discriminator", reuse=True) as scope:
        print("Preparing critic graph...")
        ## deal with y
        y = tf.reshape(y, [N_BATCH, 100, 100,1])
        c1 = convLayer(y, 1, 4, 7, 3, spec_norm=True, update_collection=update_collection) ## 32x32, 4 chhannels
        c1 = tf.nn.relu(c1)
        c2 = convLayer(c1, 2, 4, 5, 3, spec_norm=True, update_collection=update_collection) ## 10x10 4 channels
        c2 = tf.nn.relu(c2) 
        c3 = convLayer(c2, 3, 4,3,1, spec_norm=True, update_collection=update_collection) ## 8x8 4 channels
        c3 = tf.nn.relu(c3)
        c = tf.reshape(c3, [N_BATCH, 8, 8, 4])

		## go through additional conv layers to enforce locality in feedback
        c4 = convLayer(c, 4, 8,3,1, spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
        c4 = tf.nn.leaky_relu(c4)

        c5 = convLayer(c4, 5, 8,3,1, spec_norm=True, update_collection=update_collection, padStr="SAME") ## 8x8 8 channels
        c5 = tf.nn.leaky_relu(c5)
		## no go through dense layers
        c5 = tf.reshape(c5, [N_BATCH, 8 * 8 * 8])
		## dense layer

        d1 = denseLayer(c5, 6, 512, spec_norm=True, update_collection=update_collection)
        d1 = tf.nn.leaky_relu(d1)
        do1 = tf.layers.dropout(d1, rate=0.3, training=train)		

        d2 = denseLayer(do1, 7, 256, spec_norm=True, update_collection=update_collection)
        d2 = tf.nn.leaky_relu(d2)
		
        do2 = tf.layers.dropout(d2, rate=0.1, training=train)
		
        d3 = denseLayer(do2, 8, 1, spec_norm=True, update_collection=update_collection)
		## Reshape and output
        D = tf.reshape(d3, [N_BATCH, 1])
        return D ## no nonlinearity -- always applied in loss function

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
    ##########################################################################################
    path = "C:\\Jannes\\learnSamples\\040319_1W_0001s\\validation"
    outPath = "C:\\Jannes\\learnSamples\\040319_validation\\cGAN_forwDisc"
    restore = True### Set this True to load model from disk instead of training
    testSet = False
	##########################################################################################

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

    save_name = "HOLOGAN_forw.ckpt"
    save_string = os.path.join(outPath, save_name)

	### Hyperparameters
    tf.set_random_seed(42)
    eta_D = 0.0001
    eta_G = 0.0001
    eta_F = 0.0001

    N_BATCH = 60
    N_VALID = 100	
    N_CRITIC = 5
    N_REDRAW = 5	
    N_EPOCH = 50
    N_LAT = 64
    BETA = 0.01/64
    ALPHA = 1.0/64

	## sample size
    N_SAMPLE = maxFile - N_BATCH*N_CRITIC
    last_index  = 0
    print("Data set has length "+str(N_SAMPLE))

	### Define file load functions
    load_fourier = lambda x, nr : 1.0/100*np.squeeze(load_files(os.path.join(path, fourier_folder), nr, minFileNr+ x, indices))
    load_input = lambda x, nr : 1.0/255*np.squeeze(load_files(os.path.join(path, input_folder), nr, minFileNr + x, indices))
    load_output = lambda x, nr: 1.0/255*np.squeeze(load_files(os.path.join(path, output_folder), nr, minFileNr + x, indices))


    """ --------------- Set up the graph ---------------------------------------------------------"""	
	# Placeholder	
    is_train = tf.placeholder(dtype=tf.bool, name="is_train")
    X_REAL = tf.placeholder(dtype=tf.float32, name="X_REAL") ## Fourier input
    Z = tf.placeholder(dtype=tf.float32, name="Z") ## Latent variables
    Y = tf.placeholder(dtype=tf.float32, name="Y")
		
	# ROUTE THE TENSORS
    X_FAKE = generator(Z,Y, is_train, N_LAT, N_BATCH) ## GENERATOR GRAPH
    Y_HAT = forward(X_REAL, is_train, N_BATCH)
    Y_HAT_HAT = forward(X_FAKE, is_train, N_BATCH)
    D_REAL = discriminator(Y, is_train, N_BATCH, update_collection="NO_OPS") ## REAL CRITIC GRAPH
    D_FAKE = discriminator(Y_HAT_HAT, is_train,N_BATCH, update_collection=None)
    
	# Loss functions	
    F_loss = tf.nn.l2_loss(Y - Y_HAT)
    Y_HAT_loss = tf.nn.l2_loss(Y_HAT_HAT-Y)
    D_loss = tf.reduce_mean(tf.nn.softplus(D_FAKE) + tf.nn.softplus(-D_REAL))	
    G_loss = tf.reduce_mean(tf.nn.softplus(-D_FAKE)) #+ BETA*tf.nn.l2_loss(X_FAKE-X_REAL)+ ALPHA*Y_HAT_loss

    # Group variables
    #tvars = tf.trainable_variables()
    D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")#[var for var in tvars if 'critic' in var.name]
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")#[var for var in tvars if 'generator' in var.name]	
    F_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="forward")

    # Trainin operations
    D_solver = tf.train.AdamOptimizer(
            learning_rate=eta_D, 
            beta1=0.5, 
            beta2=0.9).minimize(D_loss, var_list=D_vars)

    G_solver = tf.train.AdamOptimizer(
            learning_rate=eta_G, 
            beta1=0.5, 
            beta2=0.9).minimize(G_loss, var_list=G_vars)

    F_solver = tf.train.AdamOptimizer(
            learning_rate=eta_F).minimize(F_loss, var_list=F_vars)
	
	# Initializer
    initializer = tf.global_variables_initializer() # get initializer   

	# Saver
    saver = tf.train.Saver()

    if testSet: 
        restore = True
    """ --------------- Session ---------------------------------------------------------------------------"""	
    with tf.Session() as sess:

        sess.run(initializer)
        if not restore :

            print("Commencing pretraining...")

            for i in range(0, N_SAMPLE, N_BATCH):
                    x = load_fourier(i, N_BATCH)
                    y = load_output(i, N_BATCH)
                    sess.run(F_solver, feed_dict={X_REAL:x, Y:y, is_train:True})
            
            print("Commencing training...")
            D_loss_array = [] # Discriminator loss
            G_loss_array = [] # Generator loss
            Y_loss_array = []
            percent = 0
            for j in range(N_EPOCH):   	
                for i in range(0, N_SAMPLE, N_CRITIC*N_BATCH):
                    ## Train generator
                    x = load_fourier(i, N_BATCH)
                    y = load_output(i, N_BATCH)
                    z = sample_Z(N_BATCH, N_LAT) 
                    sess.run(G_solver, feed_dict={X_REAL: x, Y: y, Z: z, is_train: True})
                    sess.run(F_solver, feed_dict={X_REAL:x, Y:y, is_train:True}) ## should be in the critic loop?

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
                        curr_Y_loss = sess.run(Y_HAT_loss, feed_dict={Y:y, Z:z, is_train:False, X_REAL:x})
                        curr_D_loss = sess.run(D_loss, feed_dict={X_REAL: x, Y: y, Z: z, is_train: False})
                        curr_G_loss = sess.run(G_loss, feed_dict={X_REAL: x, Y: y, Z: z, is_train: False})
                        Y_loss_array.append(curr_Y_loss)				
                        D_loss_array.append(curr_D_loss)
                        G_loss_array.append(curr_G_loss)
                        print("{} % ## D loss {:.1} | G loss {:.1} | Y loss {:.1}".format( percent, curr_D_loss, curr_G_loss, curr_Y_loss) )		
			    
            plt.figure(figsize=(8, 8))
            plt.plot(np.array(Y_loss_array), 'r-')
            #plt.plot(np.array(G_loss_array), 'b-')
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
                    x_pred = sess.run(X_FAKE, feed_dict={ Y:y, Z:z, is_train: False})        
                    y_pred = sess.run(Y_HAT_HAT,feed_dict={Y:y, X_REAL:x, Z:z, is_train: False})       
                    ## write the matrices to file
                    if testSet:
                        writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x_pred[0,:,:]))
                    else:
                        plotMatrices(x_pred[0,:,:], x[0,:,:])			
                        #writeMatrices(outPath, fileNr, np.squeeze(x_pred[0,:,:]), np.squeeze(y[0,:,:]), np.squeeze(x[0,:,:]))

            print("DONE! :)")


if __name__ == "__main__":
    main(sys.argv)
