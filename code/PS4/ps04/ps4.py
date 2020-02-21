"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os
from trackbar import display_trackbar_window, param, scale

##########################################################################################
# Experimental
##########################################################################################
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out

##########################################################################################

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=(float(1)/8))

    return sobel_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=(float(1)/8))

    return sobel_y


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    # pre-filtering
    # import pdb; pdb.set_trace()
    # img_a = cv2.medianBlur(img_a.astype(np.float32), 9)
    # img_b = cv2.medianBlur(img_b.astype(np.float32), 9)
    # gauss_k_size = 9
    # gauss_sigma = 10
    # img_a = cv2.GaussianBlur(img_a, (gauss_k_size, gauss_k_size), gauss_sigma)
    # img_b = cv2.GaussianBlur(img_b, (gauss_k_size, gauss_k_size), gauss_sigma)
    #
    # It = img_b - img_a
    # Ix = gradient_x(img_a)
    # Iy = gradient_y(img_a)
    #
    # # assert k_size % 2 == 1, "Even k_size, should be odd."
    #
    # IxIx = np.multiply(Ix, Ix)
    # IxIy = np.multiply(Ix, Iy)
    # IyIy = np.multiply(Iy, Iy)
    # IxIt = np.multiply(Ix, It)
    # IyIt = np.multiply(Iy, It)

    def draw_image(input_points):
        u, v = input_points
        u_v = quiver(u, v, scale=2.5, stride=8)

        return u_v

    def compute_values(kSize, sigmaGaus, det_scale, gauss_k_size=1, gauss_sigma=1):
        # gauss_k_size = 9
        # gauss_sigma = 10
        img_a_blur = cv2.GaussianBlur(img_a, (gauss_k_size, gauss_k_size), gauss_sigma)
        img_b_blur = cv2.GaussianBlur(img_b, (gauss_k_size, gauss_k_size), gauss_sigma)

        It = img_b_blur - img_a_blur
        Ix = gradient_x(img_a_blur)
        Iy = gradient_y(img_a_blur)

        # assert k_size % 2 == 1, "Even k_size, should be odd."

        IxIx = np.multiply(Ix, Ix)
        IxIy = np.multiply(Ix, Iy)
        IyIy = np.multiply(Iy, Iy)
        IxIt = np.multiply(Ix, It)
        IyIt = np.multiply(Iy, It)


        k_size = kSize
        sigma = sigmaGaus
        # smoothing kernel
        if k_type == 'uniform':
            kernel = np.ones((k_size, k_size))/(k_size**2)
        elif k_type == 'gaussian':
            kernel = cv2.getGaussianKernel(k_size, sigma)

        sum_IxIx = cv2.filter2D(IxIx, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IxIy = cv2.filter2D(IxIy, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IyIy = cv2.filter2D(IyIy, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IxIt = cv2.filter2D(IxIt, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IyIt = cv2.filter2D(IyIt, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

        A = np.array([[sum_IxIx, sum_IxIy],
                      [sum_IxIy, sum_IyIy]])
        A = A.transpose([2, 3, 0, 1]) # new shape: (M, N, 2, 2)

        b_ = np.array([sum_IxIt, sum_IyIt])
        b_ = -1*b_.transpose(1, 2, 0) # new shape: (M, N, 2)

        ## Trying Vectorization Technique
        det_threshold = 1 /(10**det_scale)
        det = sum_IxIx*sum_IyIy - sum_IxIy*sum_IxIy
        gt_idx = det > det_threshold
        lt_idx = det <= det_threshold

        U_tmp = (sum_IyIy * -sum_IxIt + -sum_IxIy * -sum_IyIt)
        U_tmp[gt_idx] /= det[gt_idx]
        U_tmp[lt_idx] = 0
        U = U_tmp

        V_tmp = (-sum_IxIy * -sum_IxIt + sum_IxIx * -sum_IyIt)
        V_tmp[gt_idx] /= det[gt_idx]
        V_tmp[lt_idx] = 0
        V = V_tmp

        return (U, V)

    # result = display_trackbar_window(
    #     'find_markers',
    #     draw_image,
    #     compute_values,
    #     kSize=param(100, 51, lambda x: x if x % 2 != 0 else x + 1),
    #     sigmaGaus=param(50, 10),
    #     det_scale=param(15, 8),
    #     gauss_sigma=param(50, 10),
    #     gauss_k_size=param(100, 9, lambda x: x if x % 2 != 0 else x + 1)
    # )

    return compute_values(k_size, sigma, det_scale=11)

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError
