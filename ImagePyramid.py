import os
import pdb

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy

imgdir = "./data/task1and2_hybrid_pyramid/"
imgnames = os.listdir(imgdir)

# Setting
kernel_size = 3
isNormal = True

# Create Dir
outdir = f"./ImagePyramid"
if not os.path.exists(outdir):
    os.mkdir(outdir)

ks_n_dir = os.path.join(outdir, f"ks{kernel_size}_{("n" if isNormal else "non")}")
if not os.path.exists(ks_n_dir):
    os.mkdir(ks_n_dir)

# Calculate
for imgname in imgnames:
    # fp = f'{imgdir}{imgnames[0]}'
    fp = os.path.join(imgdir, imgname)

    img = cv.imread(fp)
    grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Gassian filter
    x, y = np.mgrid[-(kernel_size // 2) : kernel_size // 2 + 1, -(kernel_size // 2) : kernel_size // 2 + 1]
    gaussian_kernel = (1 / 2 * np.pi) * np.exp(-(x**2+y**2) / 2) # sigma=1
    # Normalization
    if isNormal:
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    outputdirname = imgname.split('.')[0]
    outputdir = os.path.join(ks_n_dir, outputdirname)
    # outputdir = os.path.abspath(os.path.join(fp, 'Non'))
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    for idx in range(8):
        convolve_image = scipy.signal.convolve2d(grayImg, gaussian_kernel, mode='same', boundary='symm', fillvalue=1)

        plt.imshow(convolve_image, cmap=plt.get_cmap('gray'))

        if isNormal:
            plt.savefig(f'{outputdir}/{idx}.png')
        else:
            plt.savefig(f'{outputdir}/{idx}.png')
        
        grayImg = convolve_image[::2, ::2]