import numpy as np
import os
import matplotlib.pyplot as plt


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
	np.savetxt(pathName_pred_fourier_re, 255.0*pred_fourier_re, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_pred_fourier_im, 255.0*pred_fourier_im, fmt="%.1f", delimiter='\t', newline='\n')	
	np.savetxt(pathName_real_int, 255.0*real_int, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier_re, 255.0*real_fourier_re , fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier_im, 255.0*real_fourier_im , fmt="%.1f", delimiter='\t', newline='\n')

def plotMatrices(x_pred, x, x_predim, xin ):
  plt.subplot(2, 2, 1)
  plt.imshow(x_pred)
  plt.colorbar()
		
  plt.subplot(2, 2, 2)	
  plt.imshow(x)
  plt.colorbar()
	
  plt.subplot(2, 2, 3)	
  plt.imshow(x_predim)
  plt.colorbar()
	
  plt.subplot(2, 2, 4)	
  plt.imshow(xin)
  plt.colorbar()

  plt.show()