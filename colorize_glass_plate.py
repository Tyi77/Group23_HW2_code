import cv2
import numpy as np
import sys

def build_pyramid(image, levels):
    pyramid = [image]
    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def align_at_level(base_channel, moving_channel, max_translation):
    best_offset = (0, 0)
    best_score = -float('inf')
    base_mean = np.mean(base_channel)
    moving_mean = np.mean(moving_channel)

    base_normalized = base_channel - base_mean
    for mx in range(-max_translation, max_translation + 1):
        for my in range(-max_translation, max_translation + 1):
            shifted = np.roll(moving_channel, shift=(mx, my), axis=(0, 1))
            shifted_normalized = shifted - moving_mean

            numerator = np.sum(base_normalized * shifted_normalized)
            denominator = np.sqrt(np.sum(base_normalized ** 2) * np.sum(shifted_normalized ** 2))

            if denominator != 0:
                ncc_score = numerator / denominator
            else:
                ncc_score = -float('inf')

            if ncc_score > best_score:
                best_score = ncc_score
                best_offset = (mx, my)

    return best_offset


def pyramid_align(base_pyramid, moving_pyramid, max_t):
    levels = len(base_pyramid)
    total_mx, total_my = 0, 0

    # from the coarse to fine
    for i in range(levels-1, -1, -1):
        total_mx *= 2
        total_my *= 2
        base = base_pyramid[i]
        moving = moving_pyramid[i]
        # pass the shift to the next level
        moving_shift = np.roll(moving, shift=(total_mx, total_my), axis=(0, 1))
        mx, my = align_at_level(base, moving_shift, max_t)
        total_mx += mx
        total_my += my
        max_t = 2

    return total_mx, total_my



glass_plate_img = cv2.imread('data/task3_colorizing/emir.tif', cv2.IMREAD_GRAYSCALE)

if glass_plate_img is None:
    print("Error: Image not found or unable to load.")
    sys.exit()
 
# crop the boundary
black_pixels = np.where(glass_plate_img == glass_plate_img.min())
x_min = np.min(black_pixels[1])
x_max = np.max(black_pixels[1])
y_min = np.min(black_pixels[0])
y_max = np.max(black_pixels[0])
# glass_plate_img = glass_plate_img[y_min+1:y_max, x_min+1:x_max]
glass_plate_img = glass_plate_img[y_min+2:y_max-1, x_min+2:x_max-1]
cv2.imwrite('output/crop.jpg', glass_plate_img)

# divide into three images of channel
height = glass_plate_img.shape[0] // 3
width = glass_plate_img.shape[1]
print(glass_plate_img.shape)

blue_channel = glass_plate_img[:height, :]
green_channel = glass_plate_img[height:2*height, :]
red_channel = glass_plate_img[2*height:3*height, :]
cv2.imwrite('output/blue.jpg', blue_channel)
cv2.imwrite('output/green_origin.jpg', green_channel)
cv2.imwrite('output/red_origin.jpg', red_channel)

# build pyramids of each channel
levels = 5
blue_pyramid = build_pyramid(blue_channel, levels)
green_pyramid = build_pyramid(green_channel, levels)
red_pyramid = build_pyramid(red_channel, levels)

g_mx, g_my = pyramid_align(blue_pyramid, green_pyramid, 5)
r_mx, r_my = pyramid_align(blue_pyramid, red_pyramid, 5)
aligned_green_channel = np.roll(green_channel, shift=(g_mx, g_my), axis=(0, 1))
aligned_red_channel = np.roll(red_channel, shift=(r_mx, r_my), axis=(0, 1))

cv2.imwrite('output/blue.jpg', blue_channel)
cv2.imwrite('output/green.jpg', aligned_green_channel)
cv2.imwrite('output/red.jpg', aligned_red_channel)

rgb_image = np.dstack([aligned_red_channel, aligned_green_channel, blue_channel])

cv2.imwrite('output/rgb_image.jpg', rgb_image)
print(f'save color image into')
