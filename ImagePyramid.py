import os
import pdb

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy

def MyConvolve(img, kernel, fill_value):
    ks = kernel.shape[0]
    if ks % 2 == 0:
        raise Exception ('kernel size is not odd.')

    half_ks = ks // 2
    img_padded = np.pad(img, half_ks, 'constant', constant_values=fill_value)

    output_img = np.zeros(img_padded.shape)
    for i in range(half_ks, half_ks + img.shape[0]):
        for j in range(half_ks, half_ks + img.shape[1]):
            partial_img = img_padded[i-half_ks:i+half_ks+1, j-half_ks:j+half_ks+1]
            output_img[i, j] = np.sum(partial_img * kernel)

    return output_img[half_ks: half_ks + img.shape[0], half_ks: half_ks + img.shape[1]]
    
def MyImagePyramid(ori_img, kernel_size=5, levels=5) -> list:
    # Do the Image Pyramid
    # Gassian filter
    x, y = np.mgrid[-(kernel_size // 2) : kernel_size // 2 + 1, -(kernel_size // 2) : kernel_size // 2 + 1]
    gaussian_kernel = (1 / 2 * np.pi) * np.exp(-(x**2+y**2) / 2) # sigma=1
    # Normalization
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    out_list = [ori_img]
    for _ in range(1, levels):
        # convolve_image = scipy.signal.convolve2d(grayImg, gaussian_kernel, mode='same', boundary='symm', fillvalue=1)
        convolve_image = MyConvolve(ori_img, gaussian_kernel, fill_value=0)
        ds_image = convolve_image[::2, ::2]
        out_list.append(ds_image)
        
        ori_img = ds_image
    
    return out_list

def main(kernel_size=5, levels=5):
    imgdir = "./data/task1and2_hybrid_pyramid/"
    imgnames = os.listdir(imgdir)

    # Create Dir
    outdir = f"./ImagePyramid"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ks_n_dir = os.path.join(outdir, f"ks{kernel_size}")
    if not os.path.exists(ks_n_dir):
        os.mkdir(ks_n_dir)

    # imgnames = [imgnames[0]]

    # Do the Image Pyramid
    for imgname in imgnames:
        print(imgname)
        fp = os.path.join(imgdir, imgname)

        img = cv.imread(fp)
        grayImg = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Gassian filter
        x, y = np.mgrid[-(kernel_size // 2) : kernel_size // 2 + 1, -(kernel_size // 2) : kernel_size // 2 + 1]
        gaussian_kernel = (1 / 2 * np.pi) * np.exp(-(x**2+y**2) / 2) # sigma=1
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        outputdirname = imgname.split('.')[0]
        outputdir = os.path.join(ks_n_dir, outputdirname)
        # outputdir = os.path.abspath(os.path.join(fp, 'Non'))
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

        cv.imwrite(f'{outputdir}/0.png', grayImg)

        for idx in range(1, levels):
            # convolve_image = scipy.signal.convolve2d(grayImg, gaussian_kernel, mode='same', boundary='symm', fillvalue=1)
            convolve_image = MyConvolve(grayImg, gaussian_kernel, fill_value=0)
            ds_image = convolve_image[::2, ::2]

            # plt.imshow(ds_image, cmap=plt.get_cmap('gray'))

            # plt.savefig(f'{outputdir}/{idx}.png')
            cv.imwrite(f'{outputdir}/{idx}.png', ds_image)
            
            grayImg = ds_image

if __name__ == '__main__':
    main(kernel_size=5, levels=5)