import cv2
import numpy as np

image = cv2.imread('mydata/frieren.jpg')

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

blue_channel, green_channel, red_channel = cv2.split(image)
height, width = blue_channel.shape

black_width = 5
black_border = np.zeros((black_width, width), dtype=np.uint8)
black_side_border = np.zeros((height * 3 + black_width * 4, black_width), dtype=np.uint8)

stacked_image = np.vstack([
    black_border,
    blue_channel,
    black_border,
    green_channel,
    black_border, 
    red_channel,
    black_border  
])

stacked_image = np.hstack([black_side_border, stacked_image, black_side_border])

cv2.imwrite('frieren.jpg', stacked_image)

print("Image separation successfully.")
