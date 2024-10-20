import cv2
import numpy as np
import sys
import os
from ImagePyramid import MyImagePyramid

def build_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def align_at_level(base_channel, moving_channel, max_t):
    best_offset = (0, 0)
    best_score = -float('inf')
    base_norm = base_channel - np.mean(base_channel)
    moving_mean = np.mean(moving_channel)

    # find the best offset within max translation
    for mx in range(-max_t, max_t + 1):
        for my in range(-max_t, max_t + 1):
            shifted = np.roll(moving_channel, shift=(mx, my), axis=(0, 1))
            shifted_norm = shifted - moving_mean

            # use NCC to calculate the similarity
            upper = np.sum(base_norm * shifted_norm)
            lower = np.sqrt(np.sum(base_norm ** 2) * np.sum(shifted_norm ** 2))

            if lower != 0:
                ncc_score = upper / lower
            else:
                ncc_score = -float('inf')

            if ncc_score > best_score:
                best_score = ncc_score
                best_offset = (mx, my)

    return best_offset


def pyramid_align(base_pyramid, moving_pyramid, max_t):
    levels = len(base_pyramid)
    total_mx, total_my = 0, 0
    margin = 3

    # from the coarse to fine
    for i in range(levels-1, -1, -1):
        total_mx *= 2
        total_my *= 2
        base = base_pyramid[i]
        moving = moving_pyramid[i]

        # crop the image for better alignment
        base = base[margin:-margin, margin:-margin]
        moving = moving[margin:-margin, margin:-margin]
        margin *= 2

        # pass the shift to the next level
        moving_shift = np.roll(moving, shift=(total_mx, total_my), axis=(0, 1))
        mx, my = align_at_level(base, moving_shift, max_t)
        total_mx += mx
        total_my += my
        max_t = 2

    return total_mx, total_my


def colorize_glass_plate(input_path, output_dir):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    file_ext = os.path.splitext(input_path)[1].lower()
    glass_plate_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if glass_plate_img is None:
        print("Error: Image not found or unable to load.")
        sys.exit()
    
    # crop the boundary
    black_pixels = np.where(glass_plate_img == glass_plate_img.min())
    x_min = np.min(black_pixels[1])
    x_max = np.max(black_pixels[1])
    y_min = np.min(black_pixels[0])
    y_max = np.max(black_pixels[0])
    glass_plate_img = glass_plate_img[y_min+2:y_max-1, x_min+2:x_max-1]
    cv2.imwrite(f'{output_dir}/{file_name}_crop.jpg', glass_plate_img)

    # divide into three images of channel
    height = glass_plate_img.shape[0] // 3
    print(glass_plate_img.shape)

    blue_channel = glass_plate_img[:height, :]
    green_channel = glass_plate_img[height:2*height, :]
    red_channel = glass_plate_img[2*height:3*height, :]

    # build pyramids of each channel
    if file_ext == '.tif':
        levels = 5
    else:
        levels = 3
    # blue_pyramid = build_pyramid(blue_channel, levels)
    # green_pyramid = build_pyramid(green_channel, levels)
    # red_pyramid = build_pyramid(red_channel, levels)
    blue_pyramid = MyImagePyramid(blue_channel, levels=levels)
    green_pyramid = MyImagePyramid(green_channel, levels=levels)
    red_pyramid = MyImagePyramid(red_channel, levels=levels)
    
    for i in range(levels):
        cv2.imwrite(f'{output_dir}/image_pyramid/{file_name}_b_{i}.jpg', blue_pyramid[i])
        cv2.imwrite(f'{output_dir}/image_pyramid/{file_name}_g_{i}.jpg', green_pyramid[i])
        cv2.imwrite(f'{output_dir}/image_pyramid/{file_name}_r_{i}.jpg', red_pyramid[i])
    print('Finish building image pyramid')

    # align the three channel
    g_mx, g_my = pyramid_align(blue_pyramid, green_pyramid, 5)
    r_mx, r_my = pyramid_align(blue_pyramid, red_pyramid, 5)
    aligned_green_channel = np.roll(green_channel, shift=(g_mx, g_my), axis=(0, 1))
    aligned_red_channel = np.roll(red_channel, shift=(r_mx, r_my), axis=(0, 1))

    cv2.imwrite(f'{output_dir}/{file_name}_blue.jpg', blue_channel)
    cv2.imwrite(f'{output_dir}/{file_name}_green.jpg', aligned_green_channel)
    cv2.imwrite(f'{output_dir}/{file_name}_red.jpg', aligned_red_channel)

    # combine three channel and become RGB image
    rgb_image = np.dstack([blue_channel, aligned_green_channel, aligned_red_channel])

    cv2.imwrite(f'{output_dir}/{file_name}_rgb_image.jpg', rgb_image)
    print(f'save color image into "{output_dir}/{file_name}_rgb_image.jpg"')



# run colorize
# img_dir = "./data/task3_colorizing/"
img_dir = "./mydata/"
img_names = os.listdir(img_dir)

output_dir = f"./output/ColorizeGlassPlate"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(os.path.join(output_dir, 'image_pyramid')):
    os.mkdir(os.path.join(output_dir, 'image_pyramid'))

for img_name in img_names:
    colorize_glass_plate(os.path.join(img_dir, img_name), output_dir)