"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
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
    # image_out = np.zeros(image_in.shape)
    # cv2.normalize(image_in, image_out, alpha=scale_range[0],
    #               beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    image_out = cv2.normalize(image_in, dst=None, alpha=scale_range[0],
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


# def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1,
                  gauss_k_size=1, gauss_sigma_x=3, gauss_sigma_y=3):
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

    def draw_image(input):
        input_points = input[0]
        u, v = input_points
        u_v = quiver(u, v, scale=input[1], stride=input[2])

        return u_v

    def compute_values(kSize, sigmaGauss, use_img_smoothing=1,
                       gauss_k_size=1, gauss_sigma_x=1, gauss_sigma_y=1,
                       quiver_scale=1, quiver_stride=10):

        It = img_b - img_a
        Ix = gradient_x(img_a)
        Iy = gradient_y(img_b)

        # if use_img_smoothing:
        It = cv2.GaussianBlur(It,
                              (gauss_k_size, gauss_k_size),
                              sigmaX=gauss_sigma_x, sigmaY=gauss_sigma_y)
        Ix = cv2.GaussianBlur(Ix,
                              (gauss_k_size, gauss_k_size),
                              sigmaX=gauss_sigma_x, sigmaY=gauss_sigma_y)
        Iy = cv2.GaussianBlur(Iy,
                              (gauss_k_size, gauss_k_size),
                              sigmaX=gauss_sigma_x, sigmaY=gauss_sigma_y)

        IxIx = np.multiply(Ix, Ix)
        IxIy = np.multiply(Ix, Iy)
        IyIy = np.multiply(Iy, Iy)
        IxIt = np.multiply(Ix, It)
        IyIt = np.multiply(Iy, It)

        # smoothing kernel
        if k_type == 'uniform':
            kernel = np.ones((kSize, kSize))/(kSize**2)
        elif k_type == 'gaussian':
            kernel = cv2.getGaussianKernel(kSize, sigmaGauss)

        sum_IxIx = cv2.filter2D(IxIx, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IxIy = cv2.filter2D(IxIy, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IyIy = cv2.filter2D(IyIy, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IxIt = cv2.filter2D(IxIt, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        sum_IyIt = cv2.filter2D(IyIt, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

        ## Trying Vectorization Technique
        det_threshold = 1 /(10**15)
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

        U = cv2.GaussianBlur(U, (k_size, k_size), sigmaGauss)
        V = cv2.GaussianBlur(V, (k_size, k_size), sigmaGauss)

        return ((U, V), quiver_scale, quiver_stride)

    # result = display_trackbar_window(
    #     'part1_lk_quiver',
    #     draw_image,
    #     compute_values,
    #     kSize=param(100, 51, lambda x: x if x % 2 != 0 else x + 1),
    #     sigmaGauss=param(50, 30),
    #     use_img_smoothing=param(1, 0),
    #     gauss_k_size=param(100, 35, lambda x: x if x % 2 != 0 else x + 1),
    #     gauss_sigma_x=param(50, 24),
    #     gauss_sigma_y=param(50, 1),
    #     quiver_scale = param(30, 10, lambda x: x/10),
    #     quiver_stride = param(15, 10),
    #     k_size_=param(100, 35, lambda x: x if x % 2 != 0 else x + 1),
    # )
    #
    # print(result)

    # print(k_size, sigma, gauss_k_size, gauss_sigma_x, gauss_sigma_y)

    U, V = compute_values(k_size, sigma,
                          gauss_k_size=gauss_k_size, gauss_sigma_x=gauss_sigma_x,
                          gauss_sigma_y=gauss_sigma_y)[0]

    return U, V
# 1a 1
# {'kSize': 51, 'sigmaGauss': 30, 'use_img_smoothing': 1, 'gauss_k_size': 35, 'gauss_sigma_x': 10, 'gauss_sigma_y': 1, 'quiver_scale': 3.0, 'quiver_stride': 10}

# 1a 2
# {'kSize': 51, 'sigmaGauss': 30, 'use_img_smoothing': 1, 'gauss_k_size': 35, 'gauss_sigma_x': 15, 'gauss_sigma_y': 7, 'quiver_scale': 1.0, 'quiver_stride': 10}

# 1b 1
# {'kSize': 67, 'sigmaGauss': 29, 'use_img_smoothing': 1, 'gauss_k_size': 35, 'gauss_sigma_x': 24, 'gauss_sigma_y': 1, 'quiver_scale': 0.9, 'quiver_stride': 10}

# 1b 2
# {'kSize': 69, 'sigmaGauss': 30, 'use_img_smoothing': 1, 'gauss_k_size': 35, 'gauss_sigma_x': 24, 'gauss_sigma_y': 1, 'quiver_scale': 0.9, 'quiver_stride': 10}

# 1b 2
# {'kSize': 75, 'sigmaGauss': 30, 'use_img_smoothing': 1, 'gauss_k_size': 87, 'gauss_sigma_x': 24, 'gauss_sigma_y': 1, 'quiver_scale': 0.7, 'quiver_stride': 10}

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
    kernel =  np.array([1, 4, 6, 4, 1]) / 16

    filtered_image = cv2.sepFilter2D(np.float32(image), ddepth=-1, kernelX=kernel, kernelY=kernel)

    reduced_image = filtered_image[::2, ::2]
    half = tuple(x/2 for x in image.shape)
    # print(f'OG:{image.shape}, Half:{half}, New:{reduced_image.shape}')

    return reduced_image


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
    pyramid = []
    pyramid.append(image)

    for _ in range(levels-1):
        reduced_image = reduce_image(pyramid[-1])
        pyramid.append(reduced_image)

    return pyramid


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
    h = img_list[0].shape[0]
    w = sum([img.shape[1] for img in img_list])

    combined_img = np.zeros((h, w))

    i = 0
    j = 0
    for img in img_list:
        # print(f"Max: {img.max()}")
        norm_img = normalize_and_scale(img)
        # print(f"Max Normed: {norm_img.max()}")
        p, q = norm_img.shape
        combined_img[j:j+p, i:i+q] = normalize_and_scale(norm_img)
        # print(f'Before i:{i}, j:{j}, p:{p}, q:{q}')
        i = i + q
        # print(f'After i:{i}, j:{j}, p:{p}, q:{q}')
        # cv2.imshow(f'{p}, {q}', img)

    return combined_img.astype(np.uint8)


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
    w, h = image.shape
    tmp_expanded_image = np.zeros((w*2, h*2))
    tmp_expanded_image[::2, ::2] = image

    kernel = np.array([1, 4, 6, 4, 1]) / 8.

    expanded_image = cv2.sepFilter2D(np.float32(tmp_expanded_image), ddepth=-1,
                                         kernelX=kernel, kernelY=kernel)

    return expanded_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = list(range(len(g_pyr)))
    # print(l_pyr)
    # print('*'*80)
    for i in l_pyr[:-1]:
        # print(i)
        expanded = expand_image(g_pyr[i + 1])
        double = tuple(x*2 for x in g_pyr[i+1].shape)
        # print(f'OG:{g_pyr[i+1].shape}, Double:{double}, New:{expanded.shape}')
        h, w = g_pyr[i].shape
        l_pyr[i] = g_pyr[i] - expanded[:h, :w]

    l_pyr[-1] = g_pyr[-1]

    return l_pyr


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
    h, w = image.shape[:2]
    row_indices, col_indices = np.indices((h, w), dtype=np.float32)

    map_x = col_indices + U
    map_y = row_indices + V


    warped_image = cv2.remap(
        image.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation,
        borderMode=border_mode
    )

    return warped_image


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

    # def compute_values(levels, k_size, sigma, quiver_scale=1, quiver_stride=10):
    #                    # det_scale, gauss_k_size=1, gauss_sigma_x=1, gauss_sigma_y=1,
    def compute_values(levels, k_size, sigma,
                      gauss_k_size=1, gauss_sigma_x=1,
                      gauss_sigma_y=1,
                      quiver_scale=1, quiver_stride=10,
                      ):
        # create pyramids
        img_a_pyr = gaussian_pyramid(img_a, levels)
        img_b_pyr = gaussian_pyramid(img_b, levels)

        #0 start with smallest pyramid images and feed into lk
        level = levels - 1
        u, v = optic_flow_lk(img_a_pyr[level], img_b_pyr[level], k_size, k_type, sigma,
                             gauss_k_size=gauss_k_size, gauss_sigma_x=gauss_sigma_x,
                             gauss_sigma_y=gauss_sigma_y)

        while level > 0:
            #1 expand optic flow U, V matrices.
            u_exp = expand_image(u) * 2
            v_exp = expand_image(v) * 2

            level -= 1
            #2 Use expanded U, V to warp next level of img_a into img_b. (old_velocity)
            warped_img_b_to_img_a = warp(img_b_pyr[level], u_exp, v_exp, interpolation, border_mode)

            #3 Run warped_image and same level from img_2 into Lk
            u_corr, v_corr = optic_flow_lk(img_a_pyr[level], warped_img_b_to_img_a, k_size,
                                           k_type, sigma,
                                           gauss_k_size=gauss_k_size,
                                           gauss_sigma_x=gauss_sigma_x,
                                           gauss_sigma_y=gauss_sigma_y)

            #4 Add correction terms to old_velocity
            u = u_exp + u_corr
            v = v_exp + v_corr

            # goto #1

        return ((u, v), quiver_scale, quiver_stride)

    def draw_image(input):
        input_points = input[0]
        u, v = input_points
        u_v = quiver(u, v, scale=input[1], stride=input[2])

        return u_v

    # result = display_trackbar_window(
    #     'find_markers',
    #     draw_image,
    #     compute_values,
    #     levels=param(6, 4),
    #     k_size=param(100, 51, lambda x: x if x % 2 != 0 else x + 1),
    #     sigma=param(50, 30),
    #     gauss_k_size=param(100, 15, lambda x: x if x % 2 != 0 else x + 1),
    #     gauss_sigma_x=param(50, 24),
    #     gauss_sigma_y=param(50, 1),
    #     quiver_scale=param(30, 10, lambda x: x/10),
    #     quiver_stride=param(15, 10)
    # )
    # print(result)
    # return compute_values(levels, k_size, sigma ,gauss_k_size=19, gauss_sigma_x=13,
    #                   gauss_sigma_y=1)[0]
    return compute_values(levels, k_size, sigma ,gauss_k_size=11, gauss_sigma_x=19,
                      gauss_sigma_y=1)[0]

# 4a
# {'levels': 2, 'k_size': 45, 'sigma': 11, 'quiver_scale': 1.0, 'quiver_stride': 10}
# {'levels': 3, 'k_size': 49, 'sigma': 9, 'quiver_scale': 0.4, 'quiver_stride': 10}
# {'levels': 3, 'k_size': 29, 'sigma': 28, 'quiver_scale': 0.3, 'quiver_stride': 10}

# 5a
# {'levels': 2, 'k_size': 41, 'sigma': 21, 'gauss_k_size': 19, 'gauss_sigma_x': 13, 'gauss_sigma_y': 1, 'quiver_scale': 1.0, 'quiver_stride': 10}