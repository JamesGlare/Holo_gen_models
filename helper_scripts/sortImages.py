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
path = "/home/james/current_models/cVAE"
_TEXT = False	## label images
###############################################################################
class overviewImage:

	def __init__(self, text=False, printHolo=False):
		self.print_holo = printHolo
		self.text = text
		self.margin_x = 5
		self.margin_y = 5
		self.pic_x = 200
		self.pic_y = 200
		self.fourier_x = 200
		self.fourier_y = 200
		
	def __totalY(self):
		return 3*self.margin_y+2*self.pic_y

	def __totalX(self):
		if self.print_holo:
			return 4*self.margin_x+self.fourier_x+2*self.pic_x
		else:
			return 3*self.margin_x+self.fourier_x+self.pic_x

	def create_overview_image(self, int_pred, int_real, holo_pred, holo_real, fourier_pred_abs, fourier_pred_phase, fourier_real_abs, fourier_real_phase):
		out_image = Image.new('RGB', (self.__totalX(), self.__totalY()))
		out_image = self.__whiten(out_image)
		## the paste command needs the upper left corner (x,y)	

		## (1) Resize the fourier images
		fourier_pred_abs_resized = self.__rescale_no_blur(fourier_pred_abs, self.fourier_x) #fourier_pred.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)
		fourier_real_abs_resized = self.__rescale_no_blur(fourier_real_abs, self.fourier_x) #fourier_real.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)
		fourier_pred_phase_resized = self.__rescale_no_blur(fourier_pred_phase, self.fourier_x) #fourier_pred.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)
		fourier_real_phase_resized = self.__rescale_no_blur(fourier_real_phase, self.fourier_x) #fourier_real.resize((self.fourier_x, self.fourier_y), Image.ANTIALIAS)

		## rescale
		fourier_pred = self.__apply_phase_red(fourier_pred_abs_resized, fourier_pred_phase_resized)
		fourier_real = self.__apply_phase_red(fourier_real_abs_resized, fourier_real_phase_resized)
		
		## (1.1) Paste the fourier prediction image
		out_image.paste(fourier_pred, (self.margin_x, self.margin_y))
		
		## (1.2) Paste the fourier real image
		out_image.paste(fourier_real, (self.margin_x, 2*self.margin_y + self.pic_y))

		## (2) paste the holo pred image
		if self.print_holo:
			out_image.paste(holo_pred, (2*self.margin_x+self.fourier_x, self.margin_y))
		
		## (3) paste the holo real image
		if self.print_holo:
			out_image.paste(holo_real, (2*self.margin_x+self.fourier_x, 2*self.margin_y + self.pic_y))

		## (4) Now, deal with the intensity images which should be scaled up!
		int_pred_resized = int_pred.resize((self.pic_x, self.pic_y), Image.ANTIALIAS)
		int_real_resized = int_real.resize((self.pic_x, self.pic_y), Image.ANTIALIAS)
		## (4.1) Convert RGB
		int_pred_resized = self.__colorize(int_pred_resized)
		int_real_resized = self.__colorize(int_real_resized)

		## (4.2) paste the int pred image
		if self.print_holo:
			x_pos = 3*self.margin_x + self.fourier_x+ self.pic_x
		else:
			x_pos = 2*self.margin_x + self.fourier_x
				
		out_image.paste(int_pred_resized, (x_pos, self.margin_y))		
		## (4.3) paste the int real image		
		out_image.paste(int_real_resized, (x_pos, 2*self.margin_y + self.pic_y))
		
		## OPTIONAL WRITE ON IMAGE
		if self.text:
			draw_img = ImageDraw.Draw(out_image)
			draw_img.text((self.margin_x, self.margin_y), "Predicted f-matrix", fill=(235,235,235))
			draw_img.text((self.margin_x, 2*self.margin_y + self.pic_y), "Original f-matrix", fill=(235,235,235))
			draw_img.text((x_pos, self.margin_y), "Inverse intensity", fill=(235,235,235))
			draw_img.text((x_pos, 2*self.margin_y + self.pic_y), "Original intensity", fill=(235,235,235))

		return out_image		

	def __whiten(self, image):
		image_map = image.load()
		width, height = image.size 		
		
		for i in range(width):
			for j in range(height):
				image_map[i,j] = (255,255,255)

		return image

	def __apply_phase_red(self, abs, phase, re_im=False):
		w, h = abs.size

		newImg = Image.new('RGBA', (w, h),'black')
		background= Image.new('RGB', (w,h), 'white')
		newPixelMap = newImg.load()
		
		pixelMap_abs = np.transpose(np.array(abs))
		pixelMap_phase = np.transpose(np.array(phase))

		if re_im:
			z = pixelMap_abs + 1j*pixelMap_phase
			pixelMap_abs = np.abs(pixelMap_abs)
			pixelMap_phase = np.angle(z)
		
		for i in range(h):
			for j in range(w):
				value = int(40.5845*pixelMap_phase[i,j])
				newPixelMap[i,j] = (value,0,0, int(255-pixelMap_abs[i,j]))

		background.paste(newImg, (0,0), newImg)

		return background

	def __colorize(self, greyImg):
		
		rgbImg = greyImg.convert("RGB") 
		pixelMap = rgbImg.load()
		w,h = rgbImg.size
		for i in range(h):
			for j in range(w):
				value = sum(pixelMap[i,j])/3
				pixelMap[i,j] = (int(value),0,0)

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
		
def openImage(fName):
	return Image.open(fName) ## catch error?

def createDir_safely(baseDir, dirName):
	fullPath = join(baseDir, dirName)
	if not exists(fullPath):
    		makedirs(fullPath)


propertyStrings = ["pred", "real", "im"]
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
test_path = join(path, "pred_fourier")
## get all the numbers of images
file_nr = []
f_names = [f_name for f_name in listdir(path) if fileType[0] in f_name and "pred_int" in f_name]

for f in f_names:
	nr = extractNr(f)
	file_nr.append(nr)			

## build file names
if file_nr:
	nrImages = len(file_nr)

	ovImage = overviewImage(text=_TEXT) ## create overview image instance

	createDir_safely(path, "overviewImages")
	curr_nr = 0
	for i in file_nr:
		## Extract the number
		print("{}th file with #{}".format(curr_nr, i))
		int_pred_im = Image.open( join(path, buildImageName("pred", "int", i, fileType[0])))
		int_real_im = Image.open( join(path, buildImageName("real", "int", i, fileType[0])))
		holo_pred_im = Image.open( join(path, buildImageName("pred", "slm", i, fileType[0])))
		holo_real_im = Image.open( join(path, buildImageName("real", "slm", i, fileType[0])))

		fourier_pred_im_abs = Image.open( join(path, buildImageName("pred", "fourier", i, fileType[0])))
		fourier_pred_im_phase = Image.open( join(path, buildImageName("pred", "fourier_im", i, fileType[0])))
		
		fourier_real_im_abs = Image.open( join(path, buildImageName("real", "fourier", i, fileType[0])))
		fourier_real_im_phase = Image.open( join(path, buildImageName("real", "fourier_im", i, fileType[0])))
		
		ovimg = ovImage.create_overview_image(int_pred_im, int_real_im, holo_pred_im, holo_real_im, fourier_pred_im_abs, fourier_pred_im_phase, fourier_real_im_abs,fourier_real_im_phase)
		ovimg.save(join(join(path, "overviewImages"), "{0:03d}ov.png".format(i)), "PNG")
		curr_nr = curr_nr +1 
else: 
	print("Not the same number of images...\n")
