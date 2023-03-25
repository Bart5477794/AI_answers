"""Same as https://gitlab.tudelft.nl/ae-2224ii/week5/speech-recognition/-/blob/main/project_directories.py (original author: Bianca Giovanardi)
    """

import os

# main project directory
project_dir = os.path.dirname(__file__)

# data directory
data_dir = project_dir + '/data/'

# exercises directories
exe_1_dir =  project_dir + '/exe1/'
exe_2_dir =  project_dir + '/exe2/'
exe_3_dir =  project_dir + '/exe3/'
exe_4_dir =  project_dir + '/exe4/'

# dataset directory
dataset_dir = data_dir + 'dataset/'

# raw data directory
raw_data_dir = data_dir + 'raw_data/'

# pickle files directory
pickle_dir = project_dir + '/pickle/'

# figures directory
plot_dir = project_dir + '/plot/'
