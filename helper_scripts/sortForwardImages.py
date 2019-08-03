from __future__ import print_function
from sys import exit
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image
import numpy as np
from skimage import color
from skimage import img_as_float
import matplotlib.pyplot as plt

###############################################################################
path = r"/media/james/Jannes private/290719_testSet/models/expert"
###############################################################################

class forwardOverviewImage:
	def __init__(self):
		self.__margin_x = 5
		self.__margin_y = 5
		self.__outer_margin_x = 15
		self.__outer_margin_y = 15
		self.__pic_x = 200
		self.__pic_y = 200
		self.__fourier_x = 200
		self.__fourier_y = 200
	
	def __total_y(self):
		return self.__pic_y + 2*self.__outer_margin_y

	def __total_x(self):
		return 2*self.__pic_x + 2*self.__outer_margin_x + self.__margin_x
		
	def __colorize(self, greyImg):
		
		rgbImg = greyImg.convert("RGB") 
		pixelMap = rgbImg.load()
		w,h = rgbImg.size
		for i in range(h):
			for j in range(w):
				value = sum(pixelMap[i,j])/3
				pixelMap[i,j] = (value,0,0)

		return rgbImg
	
	def __rescale_no_blur(self, image, sizeXY):
		rescaled = Image.new('L', (sizeXY, sizeXY), 'black')
		w, h = image.size
		
		factorx = float(w)/sizeXY ## < 1
		factory = float(h)/sizeXY ## < 1
		rescaled_map = rescaled.load()
		original_map = image.load()
		#print(original_map)
		for i in range(0,sizeXY):
			for j in range(0,sizeXY):
				curr_box_x = int(factorx*j) #int(j / w)
				curr_box_y = int(factory*i) #int(i / h)
				if curr_box_x >= w:
					print(curr_box_x)
				curr_color = original_map[curr_box_y, curr_box_x]
				rescaled_map[i, j] = curr_color #(curr_color, curr_color, curr_color)
		return rescaled
		

	def __whiten(self, image):
		image_map = image.load()
		width, height = image.size 		
		
		for i in range(width):
			for j in range(height):
				image_map[i,j] = (255,255,255)

		return image

	def create_overview_image(self, int_pred, int_real, fourier_pred):
		outImage = Image.new('RGB', (self.__total_x(), self.__total_y() ), 'black')
		outImage = self.__whiten(outImage)

		## (1) rescale the fourier image
		#fourier_pred_rescaled = self.__rescale_no_blur(fourier_pred)
		## paste the image
		#outImage.paste(fourier_pred_rescaled, (self.__margin_x, self.__margin_y))

		## (2) paste the real intensity
		int_real_color = self.__colorize(int_real)
		int_real_color_resized = int_real_color.resize((self.__pic_x, self.__pic_y))

		outImage.paste(int_real_color_resized, (self.__outer_margin_x +self.__margin_x +self.__pic_x, self.__outer_margin_y))
	
		## (3) paste the predicted intensity
		int_pred_color = self.__colorize(int_pred)
		int_pred_color_resized = int_pred_color.resize((self.__pic_x, self.__pic_y))

		outImage.paste(int_pred_color_resized, (1*self.__outer_margin_x, self.__outer_margin_y))
		return outImage

def openImage(fName):
	return Image.open(fName) ## catch error?

def createDir_safely(baseDir, dirName):
	fullPath = join(baseDir, dirName)
	if not exists(fullPath):
    		makedirs(fullPath)

def buildImageName(propstr, typestr, nr, fileType):
	return propstr+"_"+typestr+str(nr)+"."+fileType

propertyStrings = ["pred", "real"]
typeStrings = ["int", "fourier"]
fileType= ["png"]
delimiter = ["_", "."]


def extractNr(fString):
	for substr in propertyStrings + typeStrings + fileType + delimiter:
		fString = fString.replace(substr, "")
	return int(fString)

def buildImagePath(propstr, typestr, nr):
	return join(propstr+"_"+typestr, str(nr)+"."+fileType)

def check_all_images_exist(nr):
	if not isfile(join(path, buildImagePath("pred", "int", nr))) \
	or not isfile(join(path, buildImagePath("pred", "fourier", nr))) \
	or not isfile(join(path, buildImagePath("real", "int", nr))):
		return False

def makeImage(npArray):
	return Image.fromarray(npArray)

## get all the numbers of images
file_nr = []
f_names = [f_name for f_name in listdir(path) if fileType[0] in f_name and "pred_int" in f_name]

for f in f_names:
	nr = extractNr(f)
	file_nr.append(nr)			

print(str(len(file_nr)) + " files detected")
print(str(path))

## build file names
if file_nr:
	nrImages = len(file_nr)

	ovImage = forwardOverviewImage() ## create overview image instance

	createDir_safely(path, "overviewImages")
	curr_nr = 0
	for i in file_nr:
		## Extract the number
		print("{}th file with #{}".format(curr_nr, i))

		int_pred_im = Image.open( join(path, buildImageName("pred", "int", i, fileType[0])))
		int_real_im = Image.open( join(path, buildImageName("real", "int", i, fileType[0])))
		fourier_pred_im = Image.open( join(path, buildImageName("pred", "fourier", i, fileType[0])))
		
		ovimg = ovImage.create_overview_image(int_pred_im, int_real_im, fourier_pred_im)
		ovimg.save(join(join(path, "overviewImages"), "{0:03d}ov.png".format(i)), "PNG")
		curr_nr = curr_nr +1 			
else: 
	print("Not the same number of images...\n")



