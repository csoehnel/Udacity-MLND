###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# May 11th, 2018
#
# This file contains the evaluation logic.
###############################################################################

import numpy as np

def getEPE(real_dsp, fake_dsp, params):
    if params['input_normalize'] == 3:
        real_dsp = np.multiply(real_dsp, params['glob_norm_dsp_max'] - params['glob_norm_dsp_min']) + \
                   params['glob_norm_dsp_min']
        fake_dsp = np.multiply(fake_dsp, params['glob_norm_dsp_max'] - params['glob_norm_dsp_min']) + \
                   params['glob_norm_dsp_min']
    return np.mean(np.abs(fake_dsp - real_dsp))

def evaluate(img, dsp, params, fn_runG):
    epe = []
    epe.extend([getEPE(dsp[i:i + 1], fn_runG([img[i:i + 1]])[0], params) for i in range(img.shape[0])])
    return np.mean(epe)
