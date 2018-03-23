###############################################################################
# Udacity Machine Learning Nanodegree
# Capstone Project
# Christoph Soehnel
# March 16th, 2018
#
# This file contains the evaluation logic.
###############################################################################

import numpy as np

def getEPE(real_dsp, fake_dsp):
    return np.mean(np.abs(fake_dsp - real_dsp))

def evaluate(img, dsp, fn_runG):
    epe = []
    epe.extend([getEPE(dsp[i:i + 1], fn_runG([img[i:i + 1]])[0]) for i in range(img.shape[0])])
    return np.mean(epe)
