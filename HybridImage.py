import os
import cv2 as cv
import numpy as np

imgdir = "./data/task1and2_hybrid_pyramid/"
imgnames = os.listdir(imgdir)

# Hybrid Image
# Setting
cutoff = 5

# Create Dir
outdir = f"./HybridImage"
if not os.path.exists(outdir):
    os.mkdir(outdir)

sigma_dir = os.path.join(outdir, f"cutoff{cutoff}")
if not os.path.exists(sigma_dir):
    os.mkdir(sigma_dir)

# Multiply the input image by (-1)^(x+y) to center the transform
def center_image(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] * (-1) ** (i + j)
    return img

# Compute Fourier transformation of input image, i.e. F(u,v).
def compute_Fourier(img):
    return np.fft.fft2(img)

# Create a filter function H(u,v) with the same size as the input image.
def create_filter(img, isHighPass):
    h, w, _ = img.shape
    H = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if isHighPass:
                H[i, j] = 1 - np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * cutoff ** 2))
            else:
                H[i, j] = np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * cutoff ** 2))
    return H

# Multiply F(u,v) by a filter function H(u,v).
def apply_filter(F, H):
    return F * H

# Compute the inverse Fourier transformation of the result.
def compute_inverse_Fourier(F):
    return np.fft.ifft2(F)

# Obtain the real part of the result.
def get_real_part(img):
    return np.real(img)

# Multiply the result by (-1)^(x+y)
def uncenter_image(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] * (-1) ** (i + j)
    return img

# Hybrid Image
def hybrid_image(img1, img2):
    hybrided_image = np.zeros_like(img1, dtype=np.float64)
    img1_H1 = np.zeros_like(img1, dtype=np.float64)
    img2_H2 = np.zeros_like(img2, dtype=np.float64)
    for channel in range(3):
        # Multiply the input image by (-1)^(x+y) to center the transform
        img1_channel = center_image(img1[:, :, channel])
        img2_channel = center_image(img2[:, :, channel])

        # Compute Fourier transformation of input image, i.e. F(u,v).
        F1 = compute_Fourier(img1_channel)
        F2 = compute_Fourier(img2_channel)

        # Create a filter function H(u,v) with the same size as the input image.
        H1 = create_filter(img1, isHighPass=True)
        H_normalized = cv.normalize(H1, None, 0, 255, cv.NORM_MINMAX)
        H_uint8 = H_normalized.astype(np.uint8)
        cv.imwrite(os.path.join(sigma_dir, f"{i}_H1.jpg"), H_uint8)

        H2 = create_filter(img2, isHighPass=False)
        H_normalized = cv.normalize(H2, None, 0, 255, cv.NORM_MINMAX)
        H_uint8 = H_normalized.astype(np.uint8)
        cv.imwrite(os.path.join(sigma_dir, f"{i}_H2.jpg"), H_uint8)

        # Multiply F(u,v) by a filter function H(u,v).
        F1_H1 = apply_filter(F1, H1)
        F2_H2 = apply_filter(F2, H2)

        # Compute the inverse Fourier transformation of the result.
        img1_H1_inv = compute_inverse_Fourier(F1_H1)
        img2_H2_inv = compute_inverse_Fourier(F2_H2)

        # Obtain the real part of the result.
        img1_H1_real = get_real_part(img1_H1_inv)
        img2_H2_real = get_real_part(img2_H2_inv)
        
        # Multiply the result by (-1)^(x+y)
        img1_H1[:, :, channel] = uncenter_image(img1_H1_real)
        img2_H2[:, :, channel] = uncenter_image(img2_H2_real)
        
        # Combine two images
        hybrided_image[:, :, channel] = img1_H1[:, :, channel] + img2_H2[:, :, channel]
    
    return hybrided_image, img1_H1, img2_H2

# Hybrid Image
for i in range(len(imgnames) // 2):
    imgs_with_i = [img for img in imgnames if img.startswith(str(i))]
    assert len(imgs_with_i) == 2
    img1 = cv.imread(os.path.join(imgdir, imgs_with_i[0])).astype(np.float64)
    img2 = cv.imread(os.path.join(imgdir, imgs_with_i[1])).astype(np.float64)

    # Resize img2 to the same size as img1
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

    hybrided_image, img1_H1, img2_H2 = hybrid_image(img1, img2)

    hybrided_image = np.clip(hybrided_image, 0, 255).astype(np.uint8)
    img1_H1 = np.clip(img1_H1, 0, 255).astype(np.uint8)
    img2_H2 = np.clip(img2_H2, 0, 255).astype(np.uint8)

    cv.imwrite(os.path.join(sigma_dir, f"{i}_hybrid.jpg"), hybrided_image)
    cv.imwrite(os.path.join(sigma_dir, f"{i}_img1_H1.jpg"), img1_H1)
    cv.imwrite(os.path.join(sigma_dir, f"{i}_img2_H2.jpg"), img2_H2)