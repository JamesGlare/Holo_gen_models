import matplotlib.pyplot as plt
import numpy as np
from os import getcwd, listdir
from os.path import isfile, join
from matplotlib import style
from matplotlib import rc
import matplotlib as mpl
""" ---------------------------------------------------------------
"""

###################################################################
path = "/media/james/Jannes private/190719_blazedGrating_phase_redraw/models/cGAN_FORWARD"
N_REDRAW = 5
###################################################################

def minErrorGen(errorList, step=5):
	index = 0
	max_index = errorList.size
	while index + step <= max_index :
		error_group = errorList[index:(index+step)]
		yield index + np.argmin(error_group) ## you can go past the yield
		index = index + step

def maxErrorGen(errorList, step=5):
	index = 0
	max_index = errorList.size
	while index + step <= max_index :
		error_group = errorList[index:(index+step)]
		yield index + np.argmax(error_group) ## you can go past the yield
		index = index + step


def allErrorGen(errorList, offset=0, step=5):
	index = 0
	max_index = errorList.size
	while index + offset < max_index:
		yield index + offset
		index = index + step

def loadFile(fname, path):
    if isfile(join(path, fname)):
        return np.loadtxt(join(path, fname))
    else:
        print("Failed to load file "+str(fname))

def rel_err(err, total):
    if total < 10.0 : ## basically zero except for few hotpixels
        if err < 200.0:
            return  0.0
        else:
            return err
    else:
        res = 100*float(err)/(total+0.01)
        return res

def if_zero_make_epsilon(arr):
    arr[ arr < 1 ] = 1
    return arr 

""" --------------------------------------------------------------------
"""
## (1) Load error file
err_file = loadFile("error.txt", path)
#image_entropies = loadFile("image_entropies.txt", path)
z_err_file = loadFile("z_error.txt", path)
#int_err_file = loadFile("int_error.txt", path)

int_err = err_file[:,0]# np.array([x for _,x in sorted(zip(image_entropies,err_file[:,0])) ]) 
I1 = err_file[:,1]#np.array([x for _,x in sorted(zip(image_entropies,err_file[:,1])) ]) 
I2 = err_file[:,2] # intensity of new spot

N_VALID = len(int_err)//N_REDRAW
indices = range(0, N_VALID)
print("N_VALID {}, N_REDRAW {}".format(N_VALID, N_REDRAW))


""" ---------------------------------------------------------------
"""
## (0) Plot realignment error

realign_err= loadFile("realign_error.txt", join(path, "../"))
averag_realign_err = 100*np.divide(realign_err[0:2500:5,0],realign_err[0:2500:5,1])
outtakes = averag_realign_err > 20.0
averag_realign_err[outtakes] = 0.0
print("realign_err {}".format(np.average(averag_realign_err)))

fig = plt.figure( dpi=150, figsize=(3, 2.2))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax1 = fig.add_subplot(1,1,1)
ax1.fill_between(indices, 0, averag_realign_err, color='darkred' ,  alpha=1)


ax1.set_ylim(0, 1)
ax1.set_xticks([0, 250, 500])
ax1.set_xlabel(r'Data set index $i$', fontsize=11) 

ax1.tick_params( direction="in", bottom=False, top=True, right=True)
ax1.set_ylabel(r'$E_\mathbf{I} $ [$\%$]', fontsize=11)
plt.show()

""" ------------------------------------------------------------------
"""

## (2) reform the intensity error
single_draw_int_err = np.zeros((N_REDRAW, N_VALID))
for n in range(0, N_REDRAW):
    single_draw_int_err[n,:] = [rel_err(int_err[i], I1[i]) for i in allErrorGen(int_err, n, step=N_REDRAW)]

min_draw_int_err = np.array([rel_err(int_err[i], I1[i]) for i in minErrorGen(int_err, step=N_REDRAW)])
max_draw_int_err = np.array([rel_err(int_err[i], I1[i]) for i in maxErrorGen(int_err, step=N_REDRAW)])

std_draw_int_err = np.std(single_draw_int_err, axis=0)
 
## there are some divisions by zero still - take them out
outtakes = min_draw_int_err > 20.0
min_draw_int_err[outtakes] = 0.0
max_draw_int_err[outtakes] = 0.0

print("Avg min I-error {0:.2f} +/-{1:.3f}".format(np.average(min_draw_int_err), np.average(std_draw_int_err)/np.sqrt(len(single_draw_int_err)/5) ))

## (3) Calculate the z-error
if N_REDRAW > 1:
    avg_draw_z_err = np.min(z_err_file, axis=1)
    std_draw_z_err = np.max(z_err_file, axis=1)
else:
    avg_draw_z_err = z_err_file
    std_draw_z_err = 0 ### broadcasting abuse
## (4) Get a histogram
n_bins = 20
full_hist_int_err = np.reshape(single_draw_int_err,(N_VALID*N_REDRAW,))
int_hist_outline, np_bins = np.histogram(full_hist_int_err, bins=np.linspace(0,5,n_bins)) ## full intensity histogram
min_hist_outline, np_bins = np.histogram(min_draw_int_err, bins=np.linspace(0,5,n_bins))  ## min-forw-intensity histogram
bin_centers = (np_bins[:-1] + np_bins[1:])/2.0

full_hist_z_err = np.reshape(z_err_file, (N_VALID*N_REDRAW,))                                          ## full z histogram
z_hist_outline, np_bins = np.histogram(full_hist_z_err, bins=np.linspace(0,5,n_bins))


## actually, get cumulative distributions maybe?
full_hist_int_cumsum = int_hist_outline.cumsum()
min_hist_outline_cumsum = min_hist_outline.cumsum()
z_hist_outline_cumsum = z_hist_outline.cumsum()
## normalize to 1
full_hist_int_cumsum = full_hist_int_cumsum/float(full_hist_int_cumsum[-1])
min_hist_outline_cumsum = min_hist_outline_cumsum/float(min_hist_outline_cumsum[-1])
z_hist_outline_cumsum = z_hist_outline_cumsum/float(z_hist_outline_cumsum[-1])


fig = plt.figure( dpi=150, figsize=(3, 2.2))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#for n in range(0, N_REDRAW):
#    plt.plot(indices, single_draw_int_err[n,:], color='b' , alpha=.1)
ax1= fig.add_subplot(2,2,1)
ax1.fill_between(indices, 0,max_draw_int_err, color='k' , linewidth=1, alpha=.3)
#plt.plot(indices, min_draw_int_err-std_draw_int_err, color='b',  linewidth=1, alpha=.3)
ax1.fill_between(indices, 0, min_draw_int_err, color='darkred' , alpha=1)
ax1.set_ylim(0, 5)
ax1.set_xticks([0, 250, 500])

ax1.tick_params( direction="in", bottom=False, top=True, right=True)
ax1.set_ylabel(r'$E_\mathbf{I} $ [$\%$]', fontsize=11)
ax1.set_xticklabels(["", "", ""])
ax2 = fig.add_subplot(2,2, 3)
ax2.fill_between(indices, -avg_draw_z_err,0, color='k' , alpha=1)
ax2.fill_between(indices, -std_draw_z_err, 0, color='k' , linewidth=1, alpha=.3)
ax2.set_xlabel(r'Data set index $i$', fontsize=11) 
ax2.set_xticks([0, 250, 500])
ax2.tick_params( direction="in", top=False, right=True)
ax2.set_ylabel(r'$ E_\mathbf{f} $ [$\%$]', fontsize=11)
ax2.set_ylim(-5,0)

ax3 = fig.add_subplot(1,2,2)

ax3.hist(full_hist_z_err, bins=np.linspace(0,5,n_bins), color='k' , alpha=.9, normed=1 )
ax3.hist(min_draw_int_err, bins=np.linspace(0,5,n_bins), color='darkred' , alpha=.8, normed=1 )
ax3.set_ylim(0, 1.25)
#ax3.plot(bin_centers,z_hist_outline_cumsum, color='k', alpha=1, linewidth=1.2,  )
#ax3.plot(bin_centers,full_hist_int_cumsum, color='r', alpha=1, linewidth=1.2,  )
ax3.legend([r"$\mathbf{f}$-error", r"$\mathbf{I}$-error"], edgecolor="None")
ax3.tick_params( direction="in", top=True, right=True)
ax3.set_xlabel(r'Value $E$ [$\%$]', fontsize=11) 
ax3.set_ylabel(r'Error distribution $P_{E}$', fontsize=11) 
ax3.set_yticklabels(["0","", ".5", "", "1", ""])
ax3.set_xticks([0, 2.5, 5])

plt.subplots_adjust(left=0.2, right=0.99, bottom=0.2, hspace=0, wspace=0.45)

## do the magic
plt.show()
