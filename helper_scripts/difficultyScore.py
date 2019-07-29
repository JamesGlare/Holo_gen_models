### Calculate difficulty score that allows me to sort the result images
from os.path import join
from os import listdir
import numpy as np
from math import log
#######################################################################
path = r"C:\Jannes\results\cVAE"
#######################################################################
subpath = join(path, "real_int")

def get_file_indices(path):
    indices = [f_name for f_name in listdir(path) if "txt" in f_name]
    return indices

def entropy(im):
    n,m = im.shape
    #print("Shape found {} {}".format(n, m))
    result = 0
    for i in range(0, n):
        for j in range(0, m):
            result += im[i,j]*log(1.0+im[i,j])

    return result
""" ---------------------------------------------------------------------
"""
indices = get_file_indices(subpath)
entropies = np.zeros((len(indices),1))
print("Found {} files".format(len(indices)))
nr = 0
for index in indices:
    image = 1.0/255*np.loadtxt(join(subpath, index), delimiter="\t", unpack=False)
    entropies[nr]= entropy(image)
    nr+=1

np.savetxt(join(path,"image_entropies.txt"), entropies, delimiter="\t", fmt="%.2f")