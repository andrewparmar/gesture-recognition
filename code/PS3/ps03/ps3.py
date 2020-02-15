"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


##########################################################################################
# Helper Functions
##########################################################################################
def _harris_corners(image):
    """
    Based on examples from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """
    harris_image = np.float32(image)

    dst = cv2.cornerHarris(harris_image, 9, 9, 0.180)

    results = np.where(dst > 0.1 * dst.max())

    markers = []
    for marker in zip(*results[::-1]):
        x = marker[0]
        y = marker[1]
        markers.append((x, y))

    return markers


def _cluster_markers(markers):
    """
    Based on examples from https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
    """
    markers_float32 = np.array(markers, dtype='float32')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5)

    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(markers_float32, 4, 10, criteria, 20, flags)

    markers_clustered = [tuple(x) for x in centers.astype(np.uint16).tolist()]

    return markers_clustered


def _order_markers(markers):
    """
    Input markers have the format (x, y).
    """
    if len(markers) < 4:
        return markers

    markers.sort(key=lambda x: x[0])
    left = markers[:2]
    left.sort(key=lambda x: x[1])
    right = markers[2:]
    right.sort(key=lambda x: x[1])

    top_left, bottom_left = left
    top_right, bottom_right = right

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
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    top_left = (0, 0)
    top_right = ( w -1, 0)
    bottom_left = (0, h - 1)
    bottom_right = (w - 1, h - 1)

    return [top_left, bottom_left, top_right, bottom_right]


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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    markers_positions = _harris_corners(gray_image)

    markers_positions = _cluster_markers(markers_positions)

    markers_positions = _order_markers(markers_positions)

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

    cv2.line(img, tl, tr, (230, 101, 230), thickness)
    cv2.line(img, tr, br, (230, 101, 230), thickness)
    cv2.line(img, br, bl, (230, 101, 230), thickness)
    cv2.line(img, bl, tl, (230, 101, 230), thickness)

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

    Solution based on techniques shown in
    https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function
    """
    dst_img = np.copy(imageB)

    # create indices of the destination image and linearize them
    h, w = imageB.shape[:2]
    row_indices, col_indices = np.indices((h, w), dtype=np.float32)

    # Create array where each column is a (x, y, 1) homogeneous coordinate.
    lin_homg_ind = np.array(
        [col_indices.ravel(),
         row_indices.ravel(),
         np.ones_like(col_indices).ravel()]
    )

    # Convert to inverse as we are doing backward warping.
    homography_inv = np.linalg.inv(homography)

    # Do a matrix multiplication of each homogeneous coordinate with
    #  the inverse homography.
    map_ind = homography_inv.dot(lin_homg_ind)

    # Convert to non homogeneous coordinates
    map_x, map_y = map_ind[:-1] / map_ind[-1]

    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    # Apply pixel values form imageA to imageB based on backward warping.
    dst = cv2.remap(
        imageA, map_x, map_y, cv2.INTER_LINEAR,
        dst=dst_img, borderMode=cv2.BORDER_TRANSPARENT
    )

    return dst.astype(np.uint8)


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
    sets = list(zip(src_points, dst_points))

    H = np.zeros((3, 3))
    A = np.zeros((8, 8))
    b = np.zeros(8)

    index = 0
    for i in range(0, 8, 2):
        x_s, y_s = sets[index][0]
        x_d, y_d = sets[index][1]

        A[i, :] = [x_s, y_s, 1, 0, 0, 0, -x_s*x_d, -y_s*x_d]
        A[i+1,:] = [0, 0, 0, x_s, y_s, 1, -x_s*y_d, -y_s*y_d]
        b[i] = x_d
        b[i+1] = y_d

        index += 1

    # Am = b solving for m.
    m = np.linalg.solve(A, b)

    H[0,:] = m[:3]
    H[1,:] = m[3:6]
    H[2,0:2] = m[6:9]
    H[2,2] = 1

    return H


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
