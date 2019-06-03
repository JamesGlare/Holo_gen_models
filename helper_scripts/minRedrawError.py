## Find the minimal error image in a list of images in a folder
## and using the error list

from os import listdir, makedirs
from os.path import join, isfile, exists
from shutil import copyfile
import numpy as np

#######################################################################################
path = "/media/james/SSD2_JG754/0306_inv_holo_results/030619_testSet/cWGAN"
#######################################################################################


def createDir_safely(baseDir, dirName):
	fullPath = join(baseDir, dirName)
	if not exists(fullPath):
    		makedirs(fullPath)


def toNumber(string):
	return int(string.replace("ov.png","" ))

fileNumbers = []

folder = "overviewImages"
errorFileName = "error.txt"
baseDir = join(path, folder)

# (1) get the error as a list
error = np.loadtxt(join(path, errorFileName), delimiter='\t')
error = np.squeeze(error[:,0])

N_REDRAW = 5
png_files = [ f for f in listdir(baseDir) if "png" in f]
N_FILES = len(png_files)
print("Nr of files " + str(N_FILES))

## (2) Get a list of all numbers
for f in png_files:
	fileNumbers.append(toNumber(f))
## (3) Sort that list
fileNumbers = sorted(fileNumbers)
print(len(error))
#assert(len(fileNumbers) == len(error))

N_ITER = min(len(error), N_FILES)

class groupOfFiles:
	def __init__(self, fileNumbers, error, step=5):
		self.files  = fileNumbers
		self.error = error
		self.step = step
		self.current = 0	
		self.maxiter = min(len(self.error), len(self.files))
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.current >= self.maxiter:
			raise StopIteration
		else:
			error_sub = self.error[self.current:(self.current + self.step)]
			min_index = self.current + np.argmin(error_sub)			
			self.current += self.step
			return self.files[min_index]
			
	def has_next(self):
		return self.current < self.maxiter

getMinIndices = groupOfFiles(fileNumbers, error, step=N_REDRAW)

createDir_safely(baseDir, "min_error_images")
min_error_image_dir = join(baseDir, "min_error_images")

while getMinIndices.has_next():
	file_nr = getMinIndices.next()
	src_file_name = join(baseDir, str(file_nr) +"ov.png")
	dst_file_name = join(min_error_image_dir, str(file_nr) + "ov.png")
 	copyfile(src_file_name, dst_file_name)
	print("Moving file " + str(file_nr) )

