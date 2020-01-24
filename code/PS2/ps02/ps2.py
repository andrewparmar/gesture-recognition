"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

from trackbar import display_trackbar_window, param, scale

def _get_center_and_state(img_in, circles):
    coordinate_list = circles[0].tolist()
    f = lambda x: x[1]
    coordinate_list.sort(key=f)

    states = ['red', 'yellow', 'green']

    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)

    # print("Img")
    # for circle in coordinate_list:
    #     print(img_in[circle[1], circle[0]])

    # print("Hsv")
    values = []
    for circle in coordinate_list:
        # print(img_hsv[circle[1], circle[0]])
        values.append(sum(img_hsv[circle[1], circle[0]]))

    index = np.argmax(values)

    tl_center_coordinate = tuple(coordinate_list[1][:2])
    state = states[index]
    return tl_center_coordinate, state


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    collections = []
    for radius in radii_range:
        circles = cv2.HoughCircles(img_gray,
                                   cv2.HOUGH_GRADIENT,
                                   0.5,               # inverse ratio, accumulator resolution
                                   2 * radius,      # minDist between circle centers
                                   param1=50,       # Canny edge detector upper threshold
                                   param2=8,        # Accumulator value for circle centers
                                   minRadius=radius,
                                   maxRadius=radius)
        if circles is not None:
            # print("Radius: {}, Found: {}".format(radius, len(circles[0])))

            # The accumulator returns in order of the highest votes.
            # Additional expansion could check if all three x values are equal, as traffic
            # lights are in a vertical line.
            if len(circles[0]) == 3:
                detections = circles
                collections.append(detections)
        else:
            # print("Radius: {}, Found: 0".format(radius, ))
            pass
    # import pdb; pdb.set_trace()
    # print(radius)
    # print(circles.shape)
    # print(circles)
    detections = sum(collections) / len(collections)
    detections = detections.astype(np.uint64)
    coordinates, state = _get_center_and_state(img_in, detections)

    # return coordinates, state, detections
    return coordinates, state
    # TODO: Remove the last item


# Source: https://stackoverflow.com/a/46572063
def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def find_vertices(lines):
    vertices = []
    for i in range(len(lines)):
        vertices.append(intersection(lines[i - 1], lines[i])[0])

    return vertices


def find_center(vertices, type_='hough'):
    if type_ == 'hough':
        vert = np.array(vertices)
    else:
        vert = vertices.reshape((len(vertices)*2, 2))

    row = vert[:, 1].mean(dtype=int)
    col = vert[:, 0].mean(dtype=int)

    return (col, row)


def find_polygon(lines, orientation, num_sides):
    """

    :param lines: list of lines from HoughLines()???
    :param orientation: this can be ignored for our case, (challenge?)
    :return:
    """
    '''
    valid_angles = [30, 90, 330]
    valid_angles += orientation
    
    sides = []
    for line in lines:
        if line.theta is in valid_angles:
            sides.append(lines)
            
    vertices = []
    for j in 0 to 2:
        vertices.append(intersection(sides[j-1], sides[j]))
    '''
    inner_angle = ((num_sides - 2) * np.pi ) / num_sides
    inner_degrees = inner_angle * 180 / np.pi

    for line in lines[:,0]:
        modulo = line[1] % inner_angle
        degrees = line[1] * 180 / np.pi
        print("Angle {} ({}); Inner_angle {}({}); Modulo {}".format(
            line[1], degrees, inner_angle, inner_degrees, modulo))

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)

    return edges


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    lower_red = np.array([0, 245, 245])
    upper_red = np.array([1, 255, 255])

    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img_in, img_in, mask=mask)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution, hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        res_blur = cv2.GaussianBlur(res[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = res_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLines(edges, 1, hough_angle_resolution * np.pi / 180, hough_threshold)

        # lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
        #                        hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

    def draw_image(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        find_polygon(lines, None, num_sides=3)

        for i in lines:
            rho, theta = i[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    def draw_image_hough_p(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        for i in lines:
            x1, y1, x2, y2 = i[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'yield_sign',
    #     draw_image_hough_p,
    #     compute_values,
    #     blur_sigma=param(40, 5),
    #     canny_min=param(100, 10),
    #     canny_max=param(200, 10),
    #     hough_angle_resolution=param(20, 1),
    #     hough_threshold=param(200, 80),
    #     houghP_minLineLength=param(80, 1),
    #     houghP_maxLineGap=param(50, 1)
    # )

    lines = compute_values(5, 10, 10, 6, 80, 1, 1)
    vertices = find_vertices(lines)
    center = find_center(vertices)

    return center


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([0, 245, 245])
    # upper_red = np.array([1, 255, 255])


# def normalize_hsv(values):
#
#     hue = int(round(values[0]/360.0 * 179))
#
#     saturation = int(round(values[1]/100.0 * 255))
#
#     value = int(round(values[2]/100.0 * 255))
#
#     return np.array([hue, saturation, value])


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # lower_yellow = normalize_hsv(np.array([55, 97, 99]))
    # upper_yellow = normalize_hsv(np.array([60, 98.82, 100]))
    #
    # mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # res = cv2.bitwise_and(img_in, img_in, mask=mask)

    bgr = [3, 255, 255]
    thresh = 5

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(img_in, minBGR, maxBGR)
    res = cv2.bitwise_and(img_in, img_in, mask=maskBGR)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution, hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        res_blur = cv2.GaussianBlur(res[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = res_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLines(edges, 1, hough_angle_resolution * np.pi / 180, hough_threshold)

        # lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
        #                        hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

    def draw_image(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        find_polygon(lines, None, num_sides=3)

        for i in lines:
            rho, theta = i[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    def draw_image_hough_p(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        for i in lines:
            x1, y1, x2, y2 = i[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'yield_sign',
    #     draw_image_hough_p,
    #     compute_values,
    #     blur_sigma=param(40, 5),
    #     canny_min=param(100, 10),
    #     canny_max=param(200, 10),
    #     hough_angle_resolution=param(20, 1),
    #     hough_threshold=param(200, 80),
    #     houghP_minLineLength=param(80, 1),
    #     houghP_maxLineGap=param(50, 1)
    # )

    lines = compute_values(0, 71, 49, 3, 51, 1, 1)
    # lines = compute_values(5, 10, 10, 3, 14, 37, 45)
    # lines = compute_values(10, 9, 17, 15, 9, 13, 45)
    print(len(lines))
    vertices = find_vertices(lines)
    center = find_center(vertices)

    return center


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # img_hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # lower_yellow = normalize_hsv(np.array([55, 97, 99]))
    # upper_yellow = normalize_hsv(np.array([60, 98.82, 100]))
    #
    # mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    # res = cv2.bitwise_and(img_in, img_in, mask=mask)

    bgr = [0, 127, 255]
    thresh = 5

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(img_in, minBGR, maxBGR)
    res = cv2.bitwise_and(img_in, img_in, mask=maskBGR)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution,
                       hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        res_blur = cv2.GaussianBlur(res[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = res_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        # lines = cv2.HoughLines(edges, 1, hough_angle_resolution * np.pi / 180,
        #                        hough_threshold)

        lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
                               hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

    def draw_image(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        find_polygon(lines, None, num_sides=3)

        for i in lines:
            rho, theta = i[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    def draw_image_hough_p(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        for i in lines:
            x1, y1, x2, y2 = i[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'yield_sign',
    #     draw_image_hough_p,
    #     compute_values,
    #     blur_sigma=param(40, 5),
    #     canny_min=param(100, 10),
    #     canny_max=param(200, 10),
    #     hough_angle_resolution=param(20, 1),
    #     hough_threshold=param(200, 80),
    #     houghP_minLineLength=param(80, 1),
    #     houghP_maxLineGap=param(50, 1)
    # )

    # lines = compute_values(0, 71, 49, 3, 51, 1, 1)      # HoughLines
    # vertices = compute_values(5, 10, 10, 5, 44, 73, 46)    # HoughLinesP
    vertices = compute_values(10, 10, 10, 15, 50, 0, 46)    # HoughLinesP

    # print(len(lines))
    # vertices = find_vertices(lines)
    # import pdb; pdb.set_trace()
    center = find_center(vertices, type_='hough_p')

    return center


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    hsv = cv2.cvtColor(img_in, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 245, 245])
    upper_red = np.array([1, 255, 255])


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError
