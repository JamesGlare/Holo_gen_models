from scipy.spatial import procrustes ## needed to calculate the corrected intensity error
### calculatecomplex error over data set

from os.path import join
from os import listdir
import numpy as np
from math import log
#######################################################################
path = r"/media/james/Jannes private/190719_blazedGrating_phase_redraw/models/cVAE"
#######################################################################

int_real_path = join(path, "real_int")
int_pred_path = join(path, "pred_int")

""" -------------------------------------------------------------------
"""

def get_file_indices(path):
    indices = [f_name for f_name in listdir(path) if "txt" in f_name]
    return indices

def int_error(int1, int2):
    mtx, mtx2, err = procrustes(int1, int2)
    return np.sqrt(err) ## sqrt(sum(x1-x2)^2)

""" -------------------------------------------------------------------
"""

indices = get_file_indices(int_real_path) ##
N_FILES = len(indices)
print("Found {} files".format(len(indices)))
nr = 0
int_err = np.zeros((N_FILES, 1))
for index in indices:
    int_real = np.loadtxt(join(int_real_path, index), delimiter="\t", unpack=False)
    int_pred = np.loadtxt(join(int_pred_path, index), delimiter="\t", unpack=False)

    int_err[nr] = int_error(int_real, int_pred)
    nr += 1

np.savetxt(join(path, "int_error.txt"), int_err, fmt="%.4f", delimiter="\t")
