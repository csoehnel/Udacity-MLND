import glob
from PIL import Image
from python_pfm import *

def showSampleImage():
    sample_image_path = "/home/XX/FlyingThings3D/frames_cleanpass_webp/TEST/A/0000/left/0006.webp"
    sample_dspim_path = "/home/XX/FlyingThings3D/disparity/TEST/A/0000/left/0006.pfm"
    #output_image_path = "/home/XX/img.jpg"
    #output_dimag_path = "/home/XX/dimg.jpg"

    sample_image = Image.open(sample_image_path)
    sample_image.show()
    #sample_image.save(output_image_path)

    [data, scale] = readPFM(sample_dspim_path)
    #data = (data * 255 / np.max(data)).astype('uint8') # only for saving to jpg
    disparity_image = Image.fromarray(data)
    disparity_image.show()
    #disparity_image.save(output_dimag_path)

def loadImagePaths(img_path_pattern, dsp_path_pattern):
    img_paths = glob.glob(img_path_pattern, recursive=True)
    dsp_paths = glob.glob(dsp_path_pattern, recursive=True)
    return [img_paths, dsp_paths]

def checkImagePathsConsistency(img_paths, dsp_paths):

    return [] # return list with inconsistent files

#showSampleImage()

#img_path_pattern = "/home/XX/FlyingThings3D/frames_cleanpass_webp/**/*.webp"
#dsp_path_pattern = "/home/XX/FlyingThings3D/disparity/**/*.pfm"
#[img_paths, dsp_paths] = loadImagePaths(img_path_pattern, dsp_path_pattern)

