import numpy as np
import os
import sys
import matplotlib.pyplot as plt


def writeMatrix(baseDir, iterNr, intensity):

	# build dir paths
	real_int_folder = os.path.join(baseDir, "out")
	## if directories do not exist, create them

	if not os.path.exists(real_int_folder):
		os.makedirs(real_int_folder)
	
	#build file paths
	nr_string =  '{0:05d}'.format(iterNr)
	pathName_real_int = os.path.join(real_int_folder, nr_string+".txt")

	# save matrices
	np.savetxt(pathName_real_int, intensity, fmt="%.1f", delimiter='\t', newline='\r\n')


def plot_int(matrix):
	plt.figure()
	plt.imshow(matrix)
	plt.colorbar()
	plt.show()


## produce a 100x100 image
def Gaussian(A, mu_x, mu_y, var_x, var_y, cov_x_y):
	def _gaussian(x, y): #1.0/(2*np.pi*np.sqrt(var_x+var_y))*
		dx = x-mu_x
		dy = y-mu_y
		det = var_x*var_y-cov_x_y*cov_x_y
		prod = var_y*dx*dx-2*dx*dy*cov_x_y+var_x*dy*dy
		return np.exp(-0.5*prod/det)
	
	y_vals, x_vals = np.mgrid[0:100, 0:100]
	##  find list-comprehension approach to this
	out = np.zeros((100,100))
	for j in range(0,100):
		for i in range(0,100):
			x = x_vals[i,j]
			y = y_vals[i,j]
			out[i,j] = A*_gaussian(x,y)
	return out
	

## set some parameters
max_peaks = 5
min_var = 50
max_var = 65

mu_min= 20
mu_max = 80 ## x/y asymmetry

min_int = 40
max_int = 300 # intensity budget

n_sample = 1000
outPath = "C:\\Jannes\\learnSamples\\290719_testSet_2"

if not os.path.exists(outPath):
    	os.makedirs(outPath)

for k in range(0, n_sample):
	
	sample = np.zeros((100,100))
	n_peak = int(np.random.uniform(1, max_peaks))
	curr_int = max_int/np.sqrt(n_peak)

	for i in range(0, n_peak):
		a = np.random.uniform(min_int, min(240, curr_int) )
		#curr_int = max(curr_int - a, min_int) ## reduce available intensity budget

		var_x = np.random.uniform(min_var, max_var)
		var_y = np.random.uniform(min_var, max_var)
		cov_x_y = np.random.uniform(0, max_var/3)

		mu_x = np.random.uniform(mu_min, mu_max)
		mu_y = np.random.uniform(mu_min, mu_max)
				
		sample = sample + Gaussian(a, mu_x, mu_y, var_x, var_y, cov_x_y)

	if k % 100 ==0:
		plot_int(sample)
		print(k)
	writeMatrix(outPath, k, sample)
