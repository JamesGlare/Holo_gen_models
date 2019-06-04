import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from os.path import isfile, join
from matplotlib import style

""" ----- Library ----------------------------------------------------------------- """
###############################################################################
path = "/media/james/SSD2_JG754/0306_inv_holo_results/030619_testSet/cWGAN"
n_step = 5
###############################################################################

def minErrorGen(errorList, step=5):
	index = 0
	max_index = errorList.size
	while index + step <= max_index :
		error_group = errorList[index:(index+step)]
		yield index + np.argmin(error_group) ## you can go past the yield
		index = index + step

def allErrorGen(errorList, offset=0, step=5):
	index = 0
	max_index = errorList.size
	while index + offset < max_index:
		yield index + offset
		index = index + step
		
def build_path(fName):
	cur_dir = getcwd()
	return join(cur_dir, fName)

def loadErrorFile(fName):
	if isfile(build_path(fName)):
		return np.loadtxt(build_path(fName))
	else:
		print("Failed to load file")

def plotError(errorFile, step=n_step):
	nr = (errorFile.shape)[0] ## number of rows
	
	x = np.arange(nr/step)
	err = errorFile[:,0]
	I1 = errorFile[:,1]
	I2 = errorFile[:,2]
	
	if step  == 5:
		min_err = [err[i] for i in minErrorGen(err, step)]
		print("avg forw min err = "+ str( sum(min_err)/float(len(min_err))) )
		err_0 = [err[i] for i in allErrorGen(err, 0 ,step)]
		err_1 = [err[i] for i in allErrorGen(err, 1 ,step)]
		err_2 = [err[i] for i in allErrorGen(err, 2 ,step)]
		err_3 = [err[i] for i in allErrorGen(err, 3 ,step)]
		err_4 = [err[i] for i in allErrorGen(err, 4 ,step)]
		## The single star * unpacks the sequence/collection into positional arguments, so you can do this:
		min_err, err_0, err_1, err_2, err_3, err_4 = zip(*sorted( zip(min_err, err_0, err_1, err_2, err_3, err_4), key = lambda x : x[0] ))
	else:
		print("avg forw min err = "+ str( sum(err)/float(len(err))) )
		min_err = sorted(err)
	## Plot	
	fig,ax = plt.subplots()
	if step == 5:
		plt.scatter(x, err_0, color='#CCCCCC')		
		plt.scatter(x, err_1, color='#CCCCCC')	
		plt.scatter(x, err_2, color='#CCCCCC')	
		plt.scatter(x, err_3, color='#CCCCCC')	
		plt.scatter(x, err_4, color='#CCCCCC')	
	plt.scatter(x, min_err, color='k')

	ax.tick_params(axis="x", direction="in")
	ax.tick_params(axis="y", direction="in")
	plt.ylim(0, 3500)
	plt.xlabel('Sorted validation index') 
	plt.ylabel('Intensity-reconstruction error [abs]')
	## do the magic
	plt.show()

"""----- ACTUAL SCRIPT  -----------------------------------------------"""
fName = "error.txt"
error_file = loadErrorFile(join(path, fName))
plotError(error_file, step=n_step)
