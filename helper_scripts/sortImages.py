from __future__ import print_function
from sys import exit
from os import listdir, makedirs
from os.path import isfile, join, exists
from PIL import Image, ImageDraw
import numpy as np
from skimage import color
from skimage import img_as_float
import matplotlib.pyplot as plt


###############################################################################
path = "/media/james/SSD2_JG754/0306_inv_holo_results/040319_validation/directInference/"
###############################################################################
class overviewImage:

	def __init__(self):
		self.margin_x = 20
		self.margin_y = 20
		self.pic_x = 200
		self.pic_y = 200
		self.fourier_x = 64
		self.fourier_y = 64
		
	def __totalY(self):
		return 3*self.margin_y+2*self.pic_y

	def __totalX(self):
		return 4*self.margin_x+self.fourier_x+2*self.pic_x

	def create_overview_image(self, int_pred, int_real, holo_pred, holo_real, fourier_pred, fourier_real):
		out_image = Image.new('RGB', (self.__totalX(), self.__totalY()))
		out_image = self.__whiten(out_image)
		## the paste command needs the upper left corner (x,y)	

		## (1) Resize the fourier images
		fourier_pred_resized = self.__rescale_no_blur(fourier_pred) #fourier_pred.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)
		fourier_real_resized = self.__rescale_no_blur(fourier_real) #fourier_real.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)

		## (1.1) Paste the fourier prediction image
		out_image.paste(fourier_pred_resized, (self.margin_x, self.margin_y))
		
		## (1.2) Paste the fourier real image
		out_image.paste(fourier_real_resized, (self.margin_x, 2*self.margin_y + self.pic_y))

		## (2) paste the holo pred image
		out_image.paste(holo_pred, (2*self.margin_x+self.fourier_x, self.margin_y))
		
		## (3) paste the holo real image
		out_image.paste(holo_real, (2*self.margin_x+self.fourier_x, 2*self.margin_y + self.pic_y))

		## (4) Now, deal with the intensity images which should be scaled up!
		int_pred_resized = int_pred.resize((self.pic_x, self.pic_y), Image.ANTIALIAS)
		int_real_resized = int_real.resize((self.pic_x, self.pic_y), Image.ANTIALIAS)
		## (4.1) Convert RGB
		int_pred_resized = self.__colorize(int_pred_resized)
		int_real_resized = self.__colorize(int_real_resized)

		## (4.2) paste the int pred image		
		out_image.paste(int_pred_resized, (3*self.margin_x + self.fourier_x+ self.pic_x, self.margin_y))		
		## (4.3) paste the int real image		
		out_image.paste(int_real_resized, (3*self.margin_x + self.fourier_x+ self.pic_x, 2*self.margin_y + self.pic_y))
		
		## OPTIONAL WRITE ON IMAGE
		#draw_img = ImageDraw.Draw(out_image)
		#draw_img.text((2*self.margin_x+self.fourier_x, self.margin_y), "Inv. Hologram", fill=(235,235,235))
		#draw_img.text(( 3*self.margin_x+self.fourier_x+self.pic_x,self.margin_y), "Reconstr. Intensity", fill=(235,235,235))
		#draw_img.text((2*self.margin_x+self.fourier_x,2*self.margin_y+self.pic_y), "Actual Hologram", fill=(235,235,235))
		#draw_img.text((3*self.margin_x+self.fourier_x+self.pic_x, 2*self.margin_y+self.pic_y), "Actual Intensity", fill=(235,235,235))

		return out_image		

	def __whiten(self, image):
		image_map = image.load()
		width, height = image.size 		
		
		for i in range(width):
			for j in range(height):
				image_map[i,j] = (255,255,255)

		return image
	
	def __colorize(self, greyImg):
		
		rgbImg = greyImg.convert("RGB") 
		pixelMap = rgbImg.load()
		w,h = rgbImg.size
		for i in range(h):
			for j in range(w):
				value = sum(pixelMap[i,j])/3
				pixelMap[i,j] = (int(value),0,0)

		return rgbImg
	
	def __rescale_no_blur(self, image):
		rescaled = Image.new('RGB', (self.fourier_y, self.fourier_x))
		orig_width, orig_height = image.size 		
		box_width = self.fourier_x/8
		box_height = self.fourier_y/8
		
		rescaled_map = rescaled.load()
		original_map = image.load()
		#print(original_map)
		for i in range(self.fourier_y):
			for j in range(self.fourier_x):
				curr_box_x = j / box_width
				curr_box_y = i / box_height
				curr_color = original_map[curr_box_y, curr_box_x]
				rescaled_map[i,j] = (curr_color, curr_color, curr_color)

		return rescaled
	
def openImage(fName):
	return Image.open(fName) ## catch error?

def createDir_safely(baseDir, dirName):
	fullPath = join(baseDir, dirName)
	if not exists(fullPath):
    		makedirs(fullPath)


propertyStrings = ["pred", "real"]
typeStrings = ["slm", "int", "fourier"]
fileType= ["png"]
delimiter = ["_", "."]

def extractNr(fString):
	for substr in propertyStrings + typeStrings + fileType + delimiter:
		fString = fString.replace(substr, "")
	return int(fString)

def buildImageName(propstr, typestr, nr, fileType):
	return propstr+"_"+typestr+str(nr)+"."+fileType

def check_all_images_exist(nr):
	for propstr in propertyStrings:
		for typestr in typeStrings:
			fName = buildImageName(propstr, typestr, nr, fileType[0])
			if not isfile(join(path, fName)):
				return False
	return True

### Say hello
print("Creating overview images...")
print(str(path))

## get all the numbers of images
file_nr = []
for f in listdir(path):
	## split according to string
	if fileType[0] in f: ## it's a png file !
		if "pred" in f or "real" in f:
			if "slm" in f \
			or "int" in f \
			or "fourier" in f:
				nr = extractNr(f)
				if check_all_images_exist(nr):
					file_nr.append(nr)			

## build file names
if file_nr:
	nrImages = len(file_nr)

	ovImage = overviewImage() ## create overview image instance

	createDir_safely(path, "overviewImages")
	
	for i in file_nr:
		## Extract the number

		int_pred_im = Image.open( join(path, buildImageName("pred", "int", i, fileType[0])))
		int_real_im = Image.open( join(path, buildImageName("real", "int", i, fileType[0])))
		holo_pred_im = Image.open( join(path, buildImageName("pred", "slm", i, fileType[0])))
		holo_real_im = Image.open( join(path, buildImageName("real", "slm", i, fileType[0])))
		fourier_pred_im = Image.open( join(path, buildImageName("pred", "fourier", i, fileType[0])))
		fourier_real_im = Image.open( join(path, buildImageName("real", "fourier", i, fileType[0])))
		
		ovimg = ovImage.create_overview_image(int_pred_im, int_real_im, holo_pred_im, holo_real_im, fourier_pred_im, fourier_real_im)
		ovimg.save(join(join(path, "overviewImages"), str(i)+"ov.png"), "PNG")
			
else: 
	print("Not the same number of images...\n")
