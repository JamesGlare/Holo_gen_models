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
path = "/media/james/Jannes private/290719_testSet/models/expert"
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
    if total ==0.0 :
        if err == 0.0:
            return  0.0
        else:
            return err
    else:
        return 100*float(err)/(total+0.01)

def if_zero_make_epsilon(arr):
    arr[ arr < 1 ] = 1
    return arr 

""" ---------------------------------------------------------------
"""

## (1) Load error file
err_file = loadFile("error.txt", path)
#image_entropies = loadFile("image_entropies.txt", path)

int_err = err_file[:,0] #np.array([x for _,x in sorted(zip(image_entropies,err_file[:,0])) ]) 
I1 = err_file[:,1]#np.array([x for _,x in sorted(zip(image_entropies,err_file[:,1])) ]) 
I2 = err_file[:,2] # intensity of new spot

N_VALID = len(int_err)//N_REDRAW
indices = range(0, N_VALID)
print("N_VALID {}, N_REDRAW {}".format(N_VALID, N_REDRAW))
## (2) reform the intensity error
single_draw_int_err = np.zeros((N_REDRAW, N_VALID))
for n in range(0, N_REDRAW):
    single_draw_int_err[n,:] = [rel_err(int_err[i], I1[i]) for i in allErrorGen(int_err, n, step=N_REDRAW)]

min_draw_int_err = np.array([rel_err(int_err[i], I1[i]) for i in minErrorGen(int_err, step=N_REDRAW)])
max_draw_int_err = np.array([rel_err(int_err[i], I1[i]) for i in maxErrorGen(int_err, step=N_REDRAW)])


## (4) Get a histogram
n_bins = 20
full_hist_int_err = np.reshape(single_draw_int_err,(N_VALID*N_REDRAW,))
int_hist_outline, np_bins = np.histogram(full_hist_int_err, bins=np.linspace(0,5,n_bins)) ## full intensity histogram
min_hist_outline, np_bins = np.histogram(min_draw_int_err, bins=np.linspace(0,5,n_bins))  ## min-forw-intensity histogram
bin_centers = (np_bins[:-1] + np_bins[1:])/2.0

## actually, get cumulative distributions maybe?
full_hist_int_cumsum = int_hist_outline.cumsum()
min_hist_outline_cumsum = min_hist_outline.cumsum()
## normalize to 1
full_hist_int_cumsum = full_hist_int_cumsum/float(full_hist_int_cumsum[-1])
min_hist_outline_cumsum = min_hist_outline_cumsum/float(min_hist_outline_cumsum[-1])


fig = plt.figure( dpi=150, figsize=(3, 1.5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#for n in range(0, N_REDRAW):
#    plt.plot(indices, single_draw_int_err[n,:], color='b' , alpha=.1)
ax1= fig.add_subplot(1,2,1)
ax1.fill_between(indices, 0,max_draw_int_err, color='k' , linewidth=1, alpha=.3)
#plt.plot(indices, min_draw_int_err-std_draw_int_err, color='b',  linewidth=1, alpha=.3)
ax1.fill_between(indices, 0, min_draw_int_err, color='darkred' , alpha=1)
ax1.set_ylim(0, 20)
ax1.tick_params( direction="in", bottom=False, right=True, top=True)
ax1.set_ylabel(r'$E_\mathbf{I} $ [$\%$]', fontsize=11)
ax1.set_xlabel(r'Data set index $i$', fontsize=11) 
ax1.set_xticks([0, 250, 500])
ax1.set_xticklabels(["0","250",  "500"])


ax3 = fig.add_subplot(1,2,2)

ax3.hist(min_draw_int_err, bins=np.linspace(0,20,n_bins), color='darkred' , alpha=.9, normed=1 )
#ax3.plot(bin_centers,z_hist_outline_cumsum, color='k', alpha=1, linewidth=1.2,  )
#ax3.plot(bin_centers,full_hist_int_cumsum, color='r', alpha=1, linewidth=1.2,  )
#ax3.legend([r"$\mathbf{f}$-error", r"$\mathbf{I}$-error"], edgecolor="None")
ax3.tick_params( direction="in", top=True, right=True)
ax3.set_xlabel(r'Value $E$ [$\%$]', fontsize=11) 
ax3.set_ylabel(r'$P_{E}$', fontsize=11) 
ax3.set_xticks([0, 10, 20])

#ax3.set_yticklabels(["0","", ".5", "", "1", ""])
ax3.set_ylim(0, 0.5)

plt.subplots_adjust(left=0.17, right=0.99, bottom=0.25, hspace=0, wspace=0.45)

## do the magic
plt.show()
