import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import copy as cpy
from random import shuffle	
import tensorflow as tf

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

class save_on_exit(object):
      save_if_exit = True ## static variable for save/restore control flow

      def __init__(self, sess, save_string= "model.ckpt"):
            self.save_string = save_string
            self.saver = tf.train.Saver()
            self.sess = sess

      def __enter__(self):
            save_on_exit.save_if_exit = True
            print("Save-on-exit object created")
            
            obj = cpy.copy(self)
            return obj

      def __exit__(self, type, value, traceback):
            try:
                  if save_on_exit.save_if_exit:
                        self.save_model()

            except (AttributeError, TypeError):
                  raise AssertionError("Session variable likely not valid or invalid path.")

      def restore_model(self):
            try:
                  print("Restoring the model. Hang on...")
                  save_on_exit.save_if_exit = False
                  self.saver.restore(self.sess, self.save_string)
            except (AttributeError, TypeError):
                  raise AssertionError("Save_path is None or not a valid checkpoint.")

      def save_model(self):
            self.saver.save(self.sess, self.save_string)
            save_on_exit.save_if_exit = False ## avoid saving twice
            print("Model saved in path: %s" % self.save_string)

class data_obj(object):
      
      def __init__(self, path, shuffle_data=True, test_set = False):
        self.path = path
        self.test_set = test_set ## a test set only contains output folder
        self.minFileNr = 0  ## can be used as offset if file numbers do not begin at 0
        self.re_fourier_folder = "inFourier"  ## folder in which absolute values are located (naming is historical)
        self.im_fourier_folder = "inFourierIm"  ## folder in which phase values are located
        self.input_folder = 	"in"  ## folder in which holograms are located (not used in current learning scheme)
        self.output_folder = "out"  ## folder in which intensity distributions are located
        self.indices = get_file_indices(os.path.join(path, self.output_folder))
        self.maxFile = len(self.indices) ## number of samples in data set

        if shuffle_data:
            shuffle(self.indices)  ## shuffle to get rid of possible laser/optics drift over acquisition of data set
            print("Shuffling data set...")

        self._check_folders()
	
      def _check_folders(self):
            
            if not os.path.exists(os.path.join(self.path, self.output_folder)):
                  print("Output folder not found.")
                  sys.exit()
            if not self.test_set:
                  if not os.path.exists(os.path.join(self.path, self.im_fourier_folder)):
                        print("z-phase folder not found.")
                        sys.exit()
                  elif not os.path.exists(os.path.join(self.path, self.re_fourier_folder)):
                        print("Absolute z folder not found.")
                        sys.exit()
                  else:
                        print("Data object constructed.")
            
      def _load_abs_fourier(self, x,nr) :
            return 1.0/255 * load_files(os.path.join(self.path, self.re_fourier_folder), nr, self.minFileNr+ x, self.indices)
	    
      def _load_phi_fourier(self, x, nr) : ## be careful with the argument-function: in labview, phi lives on (-pi, pi) ---- I fixed that on the labview side 
            return 1.0/(2.0*np.pi) * load_files(os.path.join(self.path, self.im_fourier_folder), nr, self.minFileNr + x, self.indices)
	
      def load_fourier(self, x, nr): 
        return np.concatenate(( np.expand_dims(self._load_abs_fourier(x,nr), axis=3), np.expand_dims(self._load_phi_fourier(x,nr), axis=3)), 3)
	
      def load_input(self,x, nr):
            return 1.0/255*np.squeeze(load_files(os.path.join(self.path, self.input_folder), nr, self.minFileNr + x, self.indices))
	    
      def load_output(self, x, nr): 
            return 1.0/255*np.squeeze(load_files(os.path.join(self.path, self.output_folder), nr, self.minFileNr + x, self.indices))

def writeMatrices(baseDir, iterNr, pred_fourier, real_int, real_fourier):
    
	assert(pred_fourier.shape == (8,8,2))
	pred_fourier_re = pred_fourier[:,:,0]
	pred_fourier_im = pred_fourier[:,:,1]
	
	assert(real_fourier.shape == (8,8,2))
	real_fourier_re = real_fourier[:,:,0]
	real_fourier_im = real_fourier[:,:,1]

	## build dir paths
	pred_fourier_re_folder = os.path.join(baseDir, "pred_fourier")
	pred_fourier_im_folder = os.path.join(baseDir, "pred_fourier_im")
	real_int_folder = os.path.join(baseDir, "real_int")
	real_fourier_re_folder = os.path.join(baseDir, "real_fourier")
	real_fourier_im_folder = os.path.join(baseDir, "real_fourier_im")

	## if directories do not exist, create them
	if not os.path.exists(pred_fourier_re_folder):
		os.makedirs(pred_fourier_re_folder)
	if not os.path.exists(pred_fourier_im_folder):
		os.makedirs(pred_fourier_im_folder)
	if not os.path.exists(real_int_folder):
		os.makedirs(real_int_folder)
	if not os.path.exists(real_fourier_re_folder):
		os.makedirs(real_fourier_re_folder)
	if not os.path.exists(real_fourier_im_folder):
    		os.makedirs(real_fourier_im_folder)

	## build file paths
	nr_string = '{0:05d}'.format(iterNr)
	pathName_pred_fourier_re = os.path.join(pred_fourier_re_folder, nr_string+".txt")
	pathName_pred_fourier_im = os.path.join(pred_fourier_im_folder, nr_string+".txt")
	pathName_real_int = os.path.join(real_int_folder, nr_string+".txt")
	pathName_real_fourier_re = os.path.join(real_fourier_re_folder, nr_string+".txt")
	pathName_real_fourier_im = os.path.join(real_fourier_im_folder, nr_string+".txt")

	## save matrices
	np.savetxt(pathName_pred_fourier_re, 255.0*pred_fourier_re, fmt="%.2f", delimiter='\t', newline='\n')
	np.savetxt(pathName_pred_fourier_im, 2.0*np.pi*pred_fourier_im, fmt="%.2f", delimiter='\t', newline='\n')	
	np.savetxt(pathName_real_int, 255.0*real_int, fmt="%.2f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier_re, 255.0*real_fourier_re , fmt="%.2f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier_im, 2.0*np.pi*real_fourier_im , fmt="%.2f", delimiter='\t', newline='\n')

def plotMatrices(x_pred, x, x_pred_phi, x_phi ):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    ax1.imshow(x_pred)
    #ax1.colorbar()
    ax1.title.set_text("r pred")
    #plt.colorbar(subplt1, cax=ax1)

    ax2 = fig.add_subplot(222)
    ax2.imshow(x)
    #ax2.colorbar()
    ax2.title.set_text("r")

    
    ax3 = fig.add_subplot(223)
    ax3.imshow(x_pred_phi)
    #ax3.colorbar()
    ax3.title.set_text("phi pred")

    
    ax4 = fig.add_subplot(224)
    ax4.imshow(x_phi)
    #ax4.colorbar()
    ax4.title.set_text("phi")

    plt.show()

def plot_forward(y, ypred):
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(y, cmap="Reds")
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(ypred, cmap="Reds")

      plt.show()