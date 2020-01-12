import cv2
import numpy as np

# Apply a Gaussian filter to remove noise
img = cv2.imread('images/saturn.png')
cv2.imshow('Img', img)
cv2.waitKey(0)

# TODO: Add noise to the image
#import pdb; pdb.set_trace()
noise_sigma = 25
noise = np.random.rand(*img.shape) * noise_sigma
noise_img = img + noise
cv2.imshow('Noisy Img', noise_img.astype(np.uint8))
cv2.waitKey(0)

# TODO: Now apply a Gaussian filter to smooth out the noise
blurred_img = cv2.GaussianBlur(img, (11, 11), 2, 2)
cv2.imshow('Blured Img', blurred_img.astype(np.uint8))
cv2.waitKey(0)
