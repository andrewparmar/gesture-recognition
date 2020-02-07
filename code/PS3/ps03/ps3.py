"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import scipy.ndimage

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


def _template_match(image, template, threshold=0.99, rotation_angle=0):
    """
    Returns list of tuples of template match coordinates.
    Coordinate are center-point of template match.
    """
    global_max = 0
    global_max_val = 0
    all_values = {}

    for theta in range(0, 360, 5):

        template_rot = scipy.ndimage.rotate(template, theta)
        threshold = 0.95

        results = cv2.matchTemplate(image, template_rot, method=cv2.TM_CCORR_NORMED)
        markers = []
        while len(markers) < 16 and threshold > 0.70:
            markers = []
            markers_ = np.where(results >= threshold)

            th, tw = template.shape

            for upper_left_pt in zip(*markers_[::-1]):
                x = upper_left_pt[0] + int(tw / 2)
                y = upper_left_pt[1] + int(th / 2)
                markers.append((x, y))

            threshold = threshold - 0.01

        if results.max() > global_max:
            print("New global Max", results.max(), "Theta: ", theta)
            global_max_val = theta
            global_max = results.max()
            all_values[theta] = markers

    markers = all_values[global_max_val]

    return markers


def _harris_corners(image, block_size=3, k_size=5, k=0.04, harris_threshold=0.99):
    # dst = cv2.cornerHarris(image, block_size, k_size, k) # (image, 3, 5, k)
    print("block_size:{}; k_size:{}; k:{}".format(block_size, k_size, k))
    dst = cv2.cornerHarris(image, 3, 5, k) # (image, 3, 5, k)

    results = np.where(dst >= harris_threshold)

    markers = []
    for marker in zip(*results[::-1]):
        x = marker[0]
        y = marker[1]
        markers.append((x, y))

    # markers = np.array(markers, dtype='float32')

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #
    # ret, label, markers_ = cv2.kmeans(markers, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # markers = [tuple(x) for x in markers.astype(np.uint8).tolist()]

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


def _color_filter_hsv(img_in, hsv, tolerance):
    """
    Inspired from https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    """
    # minHSV = np.array([hsv[0] - tolerance, hsv[1], hsv[2]])
    # maxHSV = np.array([hsv[0] + tolerance, hsv[1], hsv[2]])

    minHSV = np.array([0, 0, 0])
    maxHSV = np.array([180, 255, 50])

    maskHSV = cv2.inRange(img_in, minHSV, maxHSV)
    # resultHSV = cv2.bitwise_and(img_in, img_in, mask=maskHSV)
    return maskHSV


def _cluster_markers(markers):
    _markers = np.array(markers, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, markers_ = cv2.kmeans(_markers, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    markers_clustered = [tuple(x) for x in markers_.astype(np.uint16).tolist()]

    return markers_clustered


def _order_markers(markers):
    """

    :param markers:
    :return:
    """
    if len(markers) < 4:
        return markers

    markers.sort(key=lambda x: x[1])
    top = markers[:2]
    top.sort(key=lambda x: x[0])
    bottom = markers[2:]
    bottom.sort(key=lambda x: x[0])

    top_left, top_right = top
    bottom_left, bottom_right = bottom

    return (top_left, bottom_left, top_right, bottom_right)

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

        try:
            for marker in marker_positions:
                mark_location(new_image, marker)

            new_image = draw_box(new_image, marker_positions, 3)
        except:
            return new_image.astype(dtype=np.uint8)

        return new_image.astype(dtype=np.uint8)

    def compute_values(template_threshold, block_size, k, harris_threshold, rotation_angle):
        # Denoise while preserving edges
        denoised_image = _denoise_img(image)
        # gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # canny_template = cv2.Canny(template, 50, 200)
        # denoised_template = _denoise_img(gray_template)

        # Gray-scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray_image', gray_image)


        # Average Gray
        # img_avg = np.average(image, axis=2).astype(np.uint8)
        # threshed = cv2.adaptiveThreshold(
        #     img_avg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0
        # )

        # cv2.imshow('avg_image', threshed)
        # cv2.waitKey(0)

        # Find markers
        markers_positions = _template_match(
            gray_image, template[:, :, 1], threshold=template_threshold, rotation_angle=rotation_angle
        )
        # markers_positions = _harris_corners(
        #     gray_image, block_size, k_size= 3, k=k, harris_threshold=harris_threshold
        # )
        match_count = len(markers_positions)


        # Cluster and sort
        try:
            markers_positions = _cluster_markers(markers_positions)
            markers_positions = _order_markers(markers_positions)
        except:
            print("*"*80)
            print(markers_positions, match_count)
            print("*" * 80)

        return markers_positions

    # result = display_trackbar_window(
    #     'find_markers',
    #     draw_image,
    #     compute_values,
    #     template_threshold=param(100, 80, lambda x: x / 100),
    #     block_size=param(20, 3),
    #     k=param(3000, 2199, lambda x: x / 10000),
    #     harris_threshold=param(100, 56, lambda x: x / 100),
    #     rotation_angle=param(360, 0)
    # )

    markers_positions = compute_values(0.95, block_size=3, k=0.04, harris_threshold=0.99, rotation_angle=0)


    return markers_positions


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    tl ------------- tr
    |                 |
    |                 |
    |                 |
    bl ------------- br

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img = np.copy(image)
    tl, bl, tr, br = markers

    cv2.line(img, tl, tr, (255, 0, 0), thickness)
    cv2.line(img, tr, br, (0, 0, 255), thickness)
    cv2.line(img, br, bl, (0, 0, 0), thickness)
    cv2.line(img, bl, tl, (0, 255, 0), thickness)

    return img


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
