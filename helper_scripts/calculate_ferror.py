### calculate complex error over data set

from os.path import join
from os import listdir
import numpy as np
from math import log
#######################################################################
path = r"/media/james/Jannes private/190719_blazedGrating_phase_redraw/models/cVAE"
N_REDRAW = 5
#######################################################################

z1_abs_path = join(path, "real_fourier")
z1_phi_path = join(path, "real_fourier_im")

z2_abs_path = join(path, "pred_fourier")
z2_phi_path = join(path, "pred_fourier_im")

""" -------------------------------------------------------------------
"""

def get_file_indices(path):
    indices = [f_name for f_name in listdir(path) if "txt" in f_name]
    return indices
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

""" -------------------------------------------------------------------
"""

indices = get_file_indices(z1_abs_path) ##
N_VALID = len(indices)//N_REDRAW 
print("Found {} files".format(len(indices)))
nr = 0
z_err = np.zeros((N_VALID, N_REDRAW))
r = 0
for index in indices:
    z1_abs = np.loadtxt(join(z1_abs_path, index))
    z2_abs = np.loadtxt(join(z2_abs_path, index))
    z1_phi = np.loadtxt(join(z1_phi_path, index), delimiter="\t", unpack=False) ## don't normalize the angles
    z2_phi = np.loadtxt(join(z2_phi_path, index), delimiter="\t", unpack=False)
    
    z_err[nr, r] = np.sum(np.abs(P2R(z1_abs, z1_phi) - P2R(z2_abs, z2_phi) ))/(np.sum(np.abs(P2R(z1_abs, z1_phi))+0.01))
    r += 1
    if r % N_REDRAW == 0:
        nr +=1
        r = 0

np.savetxt(join(path, "z_error.txt"), z_err, fmt="%.4f", delimiter="\t")
