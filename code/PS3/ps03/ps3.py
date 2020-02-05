"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

##########################################################################################
# Experimental
##########################################################################################
from trackbar import display_trackbar_window, param, scale

def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)

##########################################################################################


##########################################################################################
# Helper Functions
##########################################################################################
def _denoise_img(image):
    """
    Denoise images while preserving edges.
    """
    temp_img = np.copy(image)
    d, sigma_color, sigma_space = (30, 70, 70)
    denoised_img = cv2.bilateralFilter(temp_img, d, sigma_color, sigma_space)

    return denoised_img


def _template_match(image, template, threshold=0.99):
    """
    Returns list of tuples of template match coordinates.
    Coordinate are center-point of template match.
    """
    results = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)
    markers_ = np.where(results >= threshold)

    th, tw = template.shape

    markers = []
    for upper_left_pt in zip(*markers_[::-1]):
        x = upper_left_pt[0] + int(tw / 2)
        y = upper_left_pt[1] + int(th / 2)
        markers.append((x, y))

    return markers


def _harris_corners(image, k=0.04, threshold=0.99):
    dst = cv2.cornerHarris(image, 3, 5, k)

    results = np.where(dst >= threshold)

    markers = []
    for marker in zip(*results[::-1]):
        x = marker[0]
        y = marker[1]
        markers.append((x, y))

    _markers = np.array(markers, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, markers_ = cv2.kmeans(_markers, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    markers = [tuple(x) for x in markers_.astype(np.uint8).tolist()]

    return markers


def _color_filter_bgr(img_in, bgr, tolerance):
    """
    Inspired from https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    """
    min_bgr = np.array([bgr[0] - tolerance, bgr[1] - tolerance, bgr[2] - tolerance])
    max_bgr = np.array([bgr[0] + tolerance, bgr[1] + tolerance, bgr[2] + tolerance])

    mask_bgr = cv2.inRange(img_in, min_bgr, max_bgr)
    # if bgr == [0, 0, 0]:
    #     return mask_bgr

    # img_binary = cv2.bitwise_and(img_in, img_in, mask=mask_bgr)
    return mask_bgr


##########################################################################################

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    ######################################################################################


    def draw_image(marker_positions):
        new_image = np.copy(image)

        if marker_positions is None:
            return new_image

        for marker in marker_positions:
            mark_location(new_image, marker)

        return new_image.astype(dtype=np.uint8)

    def compute_values(template_threshold, k, harris_threshold):
        # Denoise while preserving edges
        denoised_image = _denoise_img(image)

        # Mask
        bgr = [0, 0, 0]
        tolerance = 2
        img_binary = _color_filter_bgr(denoised_image, bgr, tolerance)
        img_binary = cv2.bitwise_not(img_binary)

        # Gray-scale
        # gray_image = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)

        # Fine markers
        markers_positions = _template_match(img_binary, template[:, :, 1], threshold=template_threshold)
        # markers_positions = _harris_corners(img_binary, k, harris_threshold)

        return markers_positions

    # result = display_trackbar_window(
    #     'find_markers',
    #     draw_image,
    #     compute_values,
    #     template_threshold=param(100, 50, lambda x: x / 100),
    #     k=param(3000, 2199, lambda x: x / 10000),
    #     harris_threshold=param(100, 56, lambda x: x / 100),
    # )

    # Pipeline

    # Denoise while preserving edges
    denoised_image = _denoise_img(image)

    # Mask
    bgr = [0, 0, 0]
    tolerance = 100
    img_binary = _color_filter_bgr(denoised_image, bgr, tolerance)
    img_binary = cv2.bitwise_not(img_binary)
    cv2.imshow('img_binary', img_binary.astype(np.uint8))
    cv2.waitKey(0)

    # # Gray-scale
    # gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_gray', gray_image.astype(np.uint8))
    # cv2.waitKey(0)

    # Fine markers
    markers_positions = _template_match(img_binary, template[:,:,1], threshold=0.93)
    print(markers_positions)
    return markers_positions


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError
