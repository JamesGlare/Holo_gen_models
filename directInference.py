### systemic imports
import sys	
from os import listdir, makedirs, getcwd, walk
from os.path import join, isfile, exists

### numeric imports
import numpy as np
from skimage import io
from PIL import Image				## tiff -> np array
from scipy.ndimage import gaussian_filter	## filter
from scipy.ndimage import measurements		## count particles
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt

## custom imports
from libs.input_helper import *

### library ##############################################

def argmax_2d(matrix):
	cur_max = 0
	i_m = 0 
	j_m =0 
	rows, cols = matrix.shape
	
	for i in range(rows):
		for j in range(cols):
			if matrix[i][j] > cur_max:
				cur_max = matrix[i][j]
				i_m = i 
				j_m = j
	return (i_m, j_m)

def Gaussian_8x8(A, mu_x, mu_y, var_x, var_y, cov_x_y):
	def _gaussian(x, y): #1.0/(2*np.pi*np.sqrt(var_x+var_y))*
		dx = x-mu_x
		dy = y-mu_y
		det = var_x*var_y-cov_x_y*cov_x_y
		prod = var_y*dx*dx-2*dx*dy*cov_x_y+var_x*dy*dy
		return np.exp(-0.5*prod/det)
	
	y_vals, x_vals = np.mgrid[0:8, 0:8]
	##  find list-comprehension approach to this
	out = np.zeros((8,8))
	for j in range(0,8):
		for i in range(0,8):
			x = x_vals[i,j]
			y = y_vals[i,j]
			out[i,j] = A*_gaussian(x,y)
	return out

def kronecker_delta_8x8(A, i, j):
	out = np.zeros((8,8))
	out[i,j] =A 
	return out

def flip_xy(matrix):
	return np.flip(np.flip(matrix, 0),1)

def centroid(data):
    h,w = np.shape(data)   
    x = np.arange(0,w)
    y = np.arange(0,h)

    X,Y = np.meshgrid(x,y)

    cx = np.sum(X*data)/np.sum(data)
    cy = np.sum(Y*data)/np.sum(data)

    return cx,cy


def peak_loc_nr(intensity, max_coeff=0.2):
	thrshld = max_coeff*np.max(intensity)
	intensity_thrshld = np.copy(intensity) ## advantages over assignment?
	intensity_thrshld[intensity_thrshld< thrshld] = 0
	label_img, num_obs = label(intensity_thrshld)
	peaks = find_objects(label_img) 
	centroids = []
	

	for _slice in peaks:
		dy,dx  = _slice
		x,y = dx.start, dy.start
	
		cx,cy = centroid(intensity_thrshld[_slice])
		#plot(label_img, intensity_thrshld[_slice])
		#print( str(cy) + " "+ str(cx))		
		centroids.append((x + cx,y + cy))
	
	#plot(intensity_thrshld, label_img)

	return centroids

def restrict(z, _min=0, _max=7):
	return max( min(z, _max), _min)

def random_phase(abs, phase):
    	
	for i,j in np.ndindex(abs.shape):	
		if abs[i,j] > 0.0:
    			phase[i,j] = np.random.uniform()
	return phase
"""
We have two folders path/out and path/inFourier.
 (1) Get list of all file_names from one of them
 (2) for file_name in file_list:
	int_matrix <- np.array(file_name)
	extract nr of peaks and peak locations
 (3) find a linear relation between peak positions 
	in Fourier space and real space
"""

### Change paths ######################################################
path =  r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw"
outPath = r"C:\Jannes\learnSamples\190719_blazedGrating_phase_redraw\models\expert"
N_VALID = 500
N_REDRAW = 5
testSet = False
#######################################################################

def main(argv):

	## Check PATHS
	if not exists(path):
		print("DATA SET PATH DOESN'T EXIST!")
		sys.exit()
	if not exists(outPath):
		print("MODEL/OUTPUT PATH DOESN'T EXIST!")
		sys.exit()
	## create data object
	data = data_obj(path, shuffle_data = False, test_set = testSet)
	""" Linear regression of F_p(I_p) relation
		Has to be done once in the beginning of the project. 
	"""
	"""measured = np.loadtxt(join(getcwd(), "lin_reg.txt"), delimiter='\t', skiprows=1)
	F_y = measured[:,0]
	F_x = measured[:,1]
	I_y = measured[:,2]
	I_x = measured[:,3]

	x_rel = np.polyfit(I_x, F_x, 1)
	y_rel = np.polyfit(I_y, F_y, 1)"""

	#x_rel = [ -0.20208729,  11.61005693]
	#y_rel = [ -0.15410618,  11.56273613]
	## hard copy the inferred relation
	x_rel = [ 0.1304418,  -2.8861863]		## x-relative scaling
	y_rel = [ 0.12246955,  -2.09014078]		## y-relative scaling
	A = 1.0									## coefficient to relate peak intensity to coefficients
	max_coeff = 0.3							## threshold coefficient -> multiplied with max value to obtain threshold

	mu_y = lambda cy : int(restrict(y_rel[0]*cy + y_rel[1]))
	mu_x = lambda cx : int(restrict(x_rel[0]*cx + x_rel[1]))
	

	## (2) for file_name in files:
	for nr in range(0, N_VALID):	
		
		intensity = data.load_output(nr,1)
		if not testSet:
			fourier =  data.load_fourier(nr,1)
		centroids = peak_loc_nr(intensity, max_coeff=max_coeff)

		fourier_estimate = np.zeros((8,8))
		for cx, cy in centroids:
			j = mu_x(cx)
			i = mu_y(cy)
		
			#print(str(cy) + " " + str(cx) + " -> " + str(i) + " " + str(j))
			value = float(A*intensity[int(cy), int(cx)])
			fourier_estimate[i,j] = restrict(value, _min=0.0, _max=1.0)
		
		## I don't know how to estimate phases. So, I just create an 8x8 matrix of zeros
		for r in range(0, N_REDRAW):
			file_nr = nr*N_REDRAW + r
			phase_estimate = np.zeros((8,8))
			phase_estimate = random_phase(fourier_estimate, phase_estimate)
			fourier_estimate_aug = np.concatenate((fourier_estimate[:,: , None], phase_estimate[:, :, None]), axis=2) # [8,8,2]
			## (3) plot
			if testSet:
				writeMatrices(outPath, file_nr, fourier_estimate_aug, intensity, fourier_estimate_aug)
			else:
				writeMatrices(outPath, file_nr, fourier_estimate_aug, intensity, np.squeeze(fourier)	)
		
if __name__ == "__main__":
	main(sys.argv)
