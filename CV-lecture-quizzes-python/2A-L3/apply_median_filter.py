import cv2
import numpy as np


# Helper function
def imnoise(img_in, method, dens):

    if method == 'salt & pepper':
        img_out = np.copy(img_in)
        r, c = img_in.shape
        x = np.random.rand(r, c)
        ids = x < dens / 2.
        img_out[ids] = 0
        ids = dens / 2. <= x
        ids &= x < dens
        img_out[ids] = 255

        return img_out

    else:
        print("Method {} not yet implemented.".format(method))
        exit()

# Apply a median filter

# Read an image
img = cv2.imread('images/moon.png', 0)
cv2.imshow('Image', img)

# TODO: Add salt & pepper noise
#s_vs_p = 0.5
#amount = 0.004
#out = np.copy(img)
## Salt mode
#num_salt = np.ceil(amount * img.size * s_vs_p)
#coords = [np.random.randint(0, i - 1, int(num_salt))
#      for i in img.shape]
#out[coords] = 1
#
## Pepper mode
#num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
#coords = [np.random.randint(0, i - 1, int(num_pepper))
#      for i in img.shape]
#out[coords] = 0
#
noisy_img = imnoise(img, 'salt & pepper', 0.01)
cv2.imshow('Noisy Image', noisy_img)

# TODO: Apply a median filter. Use cv2.medianBlur
cleaned_img = cv2.medianBlur(img, 3)
cv2.imshow('Cleaned Image', cleaned_img)

cv2.waitKey(0)

