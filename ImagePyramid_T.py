import os
import pdb

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy

def MyConvolve(img, kernel):
    '''
- Zero Padding
- Square kernel
'''
    padding_size = kernel.shape[0] // 2
    convole = np.zeros((img))

filefolder = "./data/task1and2_hybrid_pyramid/"
filenames = os.listdir(filefolder)

fp = f'{filefolder}{filenames[0]}'

img = cv.imread(fp)
grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# cv.imshow('Test', grayImg)
# cv.waitKey()
# print(grayImg.shape)
# pdb.set_trace()

#3*3 Gassian filter
kernel_size = 15
x, y = np.mgrid[-(kernel_size // 2) : kernel_size // 2 + 1, -(kernel_size // 2) : kernel_size // 2 + 1]
gaussian_kernel = (1 / 2 * np.pi) * np.exp(-(x**2+y**2) / 2) # sigma=1
#Normalization
# gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

# plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
# plt.colorbar()
# plt.show()

# a = np.array([[1,2,3], [1,2,3], [1,2,3]])
# b = np.array([[1,2,3]])
# a_f = np.fft.fft2(a)
# b_f = np.fft.fft2(b)
# c_f = a_f * b_f
# c = np.fft.ifft2(c_f).real
# print(c)

# print(grayImg)
scriptdir = os.path.dirname(__file__)
outputdir = os.path.abspath(os.path.join(scriptdir, 'Non'))
os.makedirs(outputdir)

for idx in range(8):
    convolve_image = scipy.signal.convolve2d(grayImg, gaussian_kernel, mode='same', boundary='symm', fillvalue=1)
    # convolve_image = MyConvolve(grayImg)
    # plt.imshow(convolve_image)
    # plt.show()

    plt.imshow(convolve_image, cmap=plt.get_cmap('gray'))
    # plt.colorbar()
    # plt.show()

    # if idx == 0:
    #     plt.savefig(f'my_image_{idx}_ks{kernel_size}.png')
    # elif idx == 1:
    #     plt.savefig(f'my_image_{idx}_ks{kernel_size}.png')
    plt.savefig(f'{outputdir}/my_image_{idx}_ks{kernel_size}.png')
    
    grayImg = convolve_image[::2, ::2]