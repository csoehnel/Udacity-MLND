###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# May 11th, 2018
#
# This is the main file for testing.
###############################################################################

from keras import backend as K
from keras.models import load_model
from datahandling import *
from evaluation import *
from vislog import *

### Params ###
params = {
    '_comment': '',
    'color': 1, # 0 = img/dsp monochrome, 1 = img color/dsp monochrome, 2 = img/dsp color
    'glob_norm_dsp_min': 1.122, # from training set
    'glob_norm_dsp_max': 951.218, # from training set
    'glob_norm_img_min': 0.0, # from training set
    'glob_norm_img_max': 255.0, # from training set
    'input_height': 256,
    'input_normalize': 3, # 0 = none, 1 = normalization, 2 = standardization, 3 = global normalization
    'input_width': 256,
    'path_generator': './logs/2018-04-19_09-28-42/generator.h5',
    'pathpattern_img': '/home/XX/FlyingThings3D/frames_cleanpass_webp/TEST/**/left/*.webp',
    'pathpattern_dsp': '/home/XX/FlyingThings3D/disparity/TEST/**/left/*.pfm',
    'removeOutliers': True,
}

# Configure KERAS to use TENSORFLOW as backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Set GPU to use (0 = first)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Save params to file
outfolder = params['path_generator'].split('/generator')[0]
with open(os.path.join(outfolder, "params_test.log"), "w") as paramfile:
    paramfile.write(repr(params))

# Clone stdout to file
sys.stdout = Logger(sys.stdout, os.path.join(outfolder, "console_test.log"))

# Configure KERAS to use TENSORFLOW as backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Set to training mode
K.set_learning_phase(1)

# Load Generator model
modelG = load_model(params['path_generator'])

# 'Real' mono image is input tensor of generator
real_img = modelG.input

# 'Fake' disparity image is output tensor of generator
fake_dsp = modelG.output

# Function for ACTUAL RUN of generator (evaluation)
fn_runG = K.function([real_img], [fake_dsp])

# Set to test mode
K.set_learning_phase(0)

### Prepare and load data ###

# Load and prepare test data
print("Preparing data...", end = " ")
t_data_start = time.time()
[test_paths, num_test] = loadFilePaths(params['pathpattern_img'], params['pathpattern_dsp'], True)
print("done after {:4.2f}s. Found {:d} test samples.".format(time.time() - t_data_start, num_test))

# Outlier outlier removal
if params['removeOutliers']:
    print("Checking for outliers...", end = " ")
    [_paths, _params] = getDatasetMinMax(test_paths, params['color'])
    test_paths = _paths
    print("done after {:4.2f}s. Removed {:d} outliers.".format(time.time() - t_data_start, num_test - len(test_paths)))
    num_test = len(test_paths)

# Preloading test data
print("Fetching test data...", end = " ")
t_data_start = time.time()
[test_img, test_dsp] = loadImageSet(test_paths, 0, num_test, params)
print("done after {:4.2f}s.".format(time.time() - t_data_start))

### Evaluation ###

print("Evaluating {:d} validation samples...".format(int(num_test)), end = " ")
t_eval_start = time.time()
epe = evaluate(test_img, test_dsp, params, fn_runG)
print("done after {:8.2f}s ==>> EPE = {:9.5f}".format(time.time() - t_eval_start, epe))
