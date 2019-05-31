from os import makedirs, getcwd, listdir, rename
from os.path import join
from random import shuffle
"""
    Script assumes that are you in a directory with folders inFourier, in, and out.
    It randomly selects N_VALID files and transfers them into a folder called validation
"""

##### EDIT THIS #####################
N_VALID = 200
#####################################

fourier_folder = "inFourier"
in_folder = "in"
out_folder = "out"

## get list of file names
file_list = listdir(join(getcwd(), fourier_folder))

file_names = [f for f in file_list if ".txt" in f]
n_files = len(file_names)
shuffle(file_names)

## create validation directory
valid_dir_name = "validation"
valid_dir_path = join(getcwd(), valid_dir_name)
valid_fourier_dir_path = join(valid_dir_path, fourier_folder)
valid_in_dir_path = join(valid_dir_path, in_folder)
valid_out_dir_path = join(valid_dir_path, out_folder)

for dir in [valid_fourier_dir_path, valid_in_dir_path, valid_out_dir_path]:
    makedirs(dir)

## Move the first N_VALID files from in, inFourier, and out 
## into the validation directory
for i in range(N_VALID):
    ## get files 
    fourier_file = join(join(getcwd(), fourier_folder), file_names[i])
    in_file = join(join(getcwd(), in_folder), file_names[i])
    out_file = join(join(getcwd(), out_folder), file_names[i])
    
    new_fourier_file = join(valid_fourier_dir_path, file_names[i])
    new_in_file = join(valid_in_dir_path, file_names[i])
    new_out_file =join(valid_out_dir_path, file_names[i])
    ## move the files
    rename(fourier_file, new_fourier_file)
    rename(in_file, new_in_file)
    rename(out_file, new_out_file)
    
print("Done")




