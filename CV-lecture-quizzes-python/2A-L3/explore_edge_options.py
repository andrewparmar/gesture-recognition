import cv2

# Explore edge options

# Load an image
img = cv2.imread('images/fall-leaves.png')
cv2.imshow('Image', img)

# TODO: Create a Gaussian filter. Use cv2.getGaussianKernel.
filter_size = 11
filter_sigma = 2

filter_kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
filter_kernel = filter_kernel * filter_kernel.T

# TODO: Apply it, specifying an edge parameter (try different parameters). Use cv2.filter2D.
edge_params = {"replicate": cv2.BORDER_CONSTANT, "symmetric": cv2.BORDER_REFLECT,
               "circular": cv2.BORDER_WRAP}

method = 'circular'
if method == 'circular':
    temp_img = cv2.copyMakeBorder(img, filter_size, filter_size, filter_size, filter_size,
                                  edge_params[method])
    smoothed = cv2.filter2D(temp_img, -1, filter_kernel)
    smoothed = smoothed[filter_size:-filter_size,
                        filter_size:-filter_size]
else:
    smoothed = cv2.filter2D(img, -1, filter_kernel, borderType=edge_params[method])

cv2.imshow('Smoothed image', smoothed)  
cv2.waitKey(0)

