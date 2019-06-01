### systemic imports
import sys	
from os import listdir, makedirs, getcwd
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

### library ##############################################

def createDir_safely( dirName):	
	if not exists(dirName):
    		makedirs(dirName)

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

def writeMatrices(baseDir, iterNr, pred_fourier, real_int, real_fourier):

	# build dir paths
	pred_fourier_folder = join(baseDir, "pred_fourier")
	real_int_folder = join(baseDir, "real_int")
	real_fourier_folder = join(baseDir, "real_fourier")

	## if directories do not exist, create them
	if not exists(pred_fourier_folder):
		makedirs(pred_fourier_folder)
	if not exists(real_int_folder):
		makedirs(real_int_folder)
	if not exists(real_fourier_folder):
		makedirs(real_fourier_folder)

	#build file paths
	nr_string = '{0:05d}'.format(iterNr)
	pathName_predFourier = join(pred_fourier_folder, nr_string+".txt")
	pathName_real_int = join(real_int_folder, nr_string+".txt")
	pathName_real_fourier= join(real_fourier_folder, nr_string+".txt")

	# save matrices
	np.savetxt(pathName_predFourier, 100.0*pred_fourier, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_int, 255.0*real_int, fmt="%.1f", delimiter='\t', newline='\n')
	np.savetxt(pathName_real_fourier, 100.0*real_fourier , fmt="%.1f", delimiter='\t', newline='\n')

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

def plot(m1, m2):
	plt.subplot(1, 2, 1)
	plt.imshow(m1)

	plt.subplot(1, 2, 2)
	plt.imshow(m2)
	plt.show()

def peak_loc_nr(intensity):
	thrshld = 0.5*np.max(intensity)
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
	

"""
We have two folders path/out and path/inFourier.
 (1) Get list of all file_names from one of them
 (2) for file_name in file_list:
	int_matrix <- np.array(file_name)
	extract nr of peaks and peak locations
 (3) find a linear relation between peak positions 
	in Fourier space and real space
"""
### sub folder name convention 
fourier_folder = "inFourier"
input_folder = 	"in"
output_folder = "out"

### Change paths ######################################################
path ="C:\\Jannes\\learnSamples\\040319_1W_0001s\\validation"
out_path =  "C:\\Jannes\\learnSamples\\040319_validation\\directInference"
N_VALID = 100
#######################################################################


def main(argv):

	""" Linear regression of F_p(I_p) relation """
	"""measured = np.loadtxt(join(getcwd(), "lin_reg.txt"), delimiter='\t', skiprows=1)
	F_y = measured[:,0]
	F_x = measured[:,1]
	I_y = measured[:,2]
	I_x = measured[:,3]

	x_rel = np.polyfit(I_x, F_x, 1)
	y_rel = np.polyfit(I_y, F_y, 1)"""

	x_rel = [ -0.20208729,  11.61005693]
	y_rel = [ -0.15410618,  11.56273613]
	
	mu_y = lambda cy : int(restrict(y_rel[0]*cy + y_rel[1]))
	mu_x = lambda cx : int(restrict(x_rel[0]*cx + x_rel[1]))
	
	## (1) Get a list of .txt files
	int_path = join(path,  output_folder)
	fourier_path = join(path, fourier_folder)
	files = [ f for f in listdir(int_path ) if ".txt" in f]
	print("Found " + str(len(files)) + " txt files.")
	
	## (2) for file_name in files:
	file_nr = 0
	for file_name in files:	
		
		intensity = 1./255.*np.loadtxt(join(int_path, file_name), delimiter='\t', unpack=False)
		fourier = 1./100.*np.loadtxt(join(fourier_path, file_name), delimiter='\t', unpack=False)
		centroids = peak_loc_nr(intensity)
		
		#plt.figure()
		#plt.imshow(intensity)
		#plt.show()

		fourier_estimate = np.zeros((8,8))
		for cx, cy in centroids:
			j = mu_x(cx)
			i = mu_y(cy)
		
			#print(str(cy) + " " + str(cx) + " -> " + str(i) + " " + str(j))
			value = float(1.0/2.5*intensity[int(cy), int(cx)])
			fourier_estimate[i,j] = restrict(value, _min=0.0, _max=1.0)
		## (3) plot
		writeMatrices(out_path, file_nr, fourier_estimate, intensity, fourier) 
		file_nr = file_nr + 1

		if file_nr == N_VALID:
			sys.exit()
			
if __name__ == "__main__":
	main(sys.argv)
