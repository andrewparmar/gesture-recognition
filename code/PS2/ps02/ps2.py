"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

from trackbar import display_trackbar_window, param


# Helper functions #######################################################################

def _add_cross_hairs(img, coordinates):
    x, y = coordinates
    color = (0, 0, 0)
    thickness = 2
    cv2.line(img, (x - 5, y), (x + 5, y), color, thickness)  # cross-hair horizontal
    cv2.line(img, (x, y + 5), (x, y - 5), color, thickness)  # cross-hair vertical

    return img


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
        # import pdb; pdb.set_trace()
        values.append(sum(img_hsv[circle[1], circle[0]]))

    index = np.argmax(values)

    tl_center_coordinate = tuple(coordinate_list[1][:2])
    state = states[index]
    return tl_center_coordinate, state


def _get_polygon_center(vertices, triangle=False):
    vertex_arr = vertices.reshape((-1, 2))

    max_x = vertex_arr[:, 0].max()
    min_x = vertex_arr[:, 0].min()
    max_y = vertex_arr[:, 1].max()
    min_y = vertex_arr[:, 1].min()

    center_x = int((min_x + max_x) / 2)
    if not triangle:
        center_y = int((min_y + max_y) / 2)
    else:
        center_y = int((min_y + min_y + max_y) / 3)

    return (center_x, center_y)


def color_filter_bgr(img_in, bgr, tolerance):
    minBGR = np.array([bgr[0] - tolerance, bgr[1] - tolerance, bgr[2] - tolerance])
    maxBGR = np.array([bgr[0] + tolerance, bgr[1] + tolerance, bgr[2] + tolerance])

    maskBGR = cv2.inRange(img_in, minBGR, maxBGR)
    img_binary = cv2.bitwise_and(img_in, img_in, mask=maskBGR)

    return img_binary


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

    vertices = np.array(vertices)
    return vertices


def find_center(vertices):
    vert = np.array(vertices)

    row = vert[:, 1].mean(dtype=int)
    col = vert[:, 0].mean(dtype=int)

    return (col, row)


##########################################################################################

def traffic_light_detection(img_in, radii_range, blackout=False):
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

    bgr = [51, 51, 51]
    tolerance = 2
    img_binary = color_filter_bgr(img_in, bgr, tolerance)
    img_binary_single_channel = img_binary[:,:,0]
    # cv2.imwrite("part1_test.png".format(), img_binary_single_channel)

    collections = []
    for radius in radii_range:
        circles = cv2.HoughCircles(img_binary_single_channel,
                                   cv2.HOUGH_GRADIENT,
                                   1,  # inverse ratio, accumulator resolution
                                   2 * radius,  # minDist between circle centers
                                   param1=50,  # Canny edge detector upper threshold
                                   param2=8,  # Accumulator value for circle centers
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
            # import pdb; pdb.set_trace()
            pass
    # import pdb; pdb.set_trace()
    # print(radius)
    # print(circles.shape)
    # print(circles)

    def compute_values(accumulator_resolution, min_dist, param1, param2, min_radius, max_radius):

        circles = cv2.HoughCircles(img_binary_single_channel,
                                   cv2.HOUGH_GRADIENT,
                                   accumulator_resolution,  # inverse ratio, accumulator resolution
                                   min_dist,  # minDist between circle centers
                                   param1=param1,  # Canny edge detector upper threshold
                                   param2=param2,  # Accumulator value for circle centers
                                   minRadius=min_radius,
                                   maxRadius=max_radius)

        return circles

    def draw_image_hough_p(circles):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if circles is None:
            return output_img
        print(len(circles[0]))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(i)

            # output_img = _add_cross_hairs(output_img, (int(i[0]), int(i[1])))

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'traffic_sign',
    #     draw_image_hough_p,
    #     compute_values,
    #     accumulator_resolution=param(3, 1),
    #     min_dist=param(80, 10),
    #     param1=param(100, 50),
    #     param2=param(20, 8),
    #     min_radius=param(40, 10),
    #     max_radius=param(40, 30),
    # )
    if len(collections) == 0:
        if blackout:
            return None, None, img_in
        else:
            return None, None

    detections = sum(collections) / len(collections)
    detections = detections.astype(np.uint64)
    coordinates, state = _get_center_and_state(img_in, detections)

    if blackout:
        for circle_coordinates in detections[0, :]:
            # import pdb; pdb.set_trace()
            x, y, r = tuple(int(i) for i in circle_coordinates)
            cv2.circle(img_in, (x, y), r+2, (0, 0, 0), -1)
        return coordinates, state, img_in

    return coordinates, state


def yield_sign_detection(img_in, blackout=False):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    bgr = [0, 0, 255]
    tolerance = 1
    img_binary = color_filter_bgr(img_in, bgr, tolerance)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution,
                       hough_threshold, houghP_minLineLength, houghP_maxLineGap):
        img_binary_blur = cv2.GaussianBlur(img_binary[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = img_binary_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLines(edges, 1, hough_angle_resolution * np.pi / 180,
                               hough_threshold)

        # lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
        #                         hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

    def draw_image(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # output_img = np.copy(img_binary)

        if lines is None:
            return output_img
        print(len(lines))

        # find_polygon(lines, None, num_sides=3)

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

        print(lines)
        try:
            vertices = find_vertices(lines)
            center = _get_polygon_center(vertices, triangle=True)
            output_img = _add_cross_hairs(output_img, center)
        except:
            print("***** Error")

        return output_img.astype(dtype=np.uint8)

    def draw_image_hough_p(lines):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if lines is None:
            return output_img
        print(len(lines))

        for i in lines:
            x1, y1, x2, y2 = i[0]
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        center = _get_polygon_center(lines, triangle=True)

        output_img = _add_cross_hairs(output_img, center)

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'yield_sign',
    #     draw_image,
    #     compute_values,
    #     blur_sigma=param(40, 3),
    #     canny_min=param(100, 10),
    #     canny_max=param(200, 10),
    #     hough_angle_resolution=param(20, 6),
    #     hough_threshold=param(200, 80),
    #     houghP_minLineLength=param(80, 1),
    #     houghP_maxLineGap=param(50, 1)
    # )

    # vertices = compute_values(3, 10, 10, 6, 27, 38, 25)
    lines = compute_values(3, 10, 10, 6, 80, 1, 1)
    # import pdb; pdb.set_trace()
    if lines is None or len(lines) < 3:
        if blackout:
            return None, img_in
        else:
            return None
    vertices = find_vertices(lines)
    center = _get_polygon_center(vertices, triangle=True)

    if blackout:
        return center, img_in

    return center


def stop_sign_detection(img_in, blackout=False):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    bgr = [0, 0, 204]
    tolerance = 5
    img_binary = color_filter_bgr(img_in, bgr, tolerance)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution,
                       hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        img_binary_blur = cv2.GaussianBlur(img_binary[:, :, 2], (3, 3), blur_sigma)

        image_for_canny = img_binary_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
                                hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

    def draw_image_hough_p(lines):
        # output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        output_img = np.copy(img_binary)

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
    #     houghP_minLineLength=param(200, 1),
    #     houghP_maxLineGap=param(50, 1)
    # )

    vertices = compute_values(5, 10, 10, 3, 18, 0, 14)

    if vertices is None:
        return None

    center = _get_polygon_center(vertices)

    if blackout:
        return center, img_in

    return center


def warning_sign_detection(img_in, blackout=False):
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
    tolerance = 5
    img_binary = color_filter_bgr(img_in, bgr, tolerance)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution,
                       hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        img_binary_blur = cv2.GaussianBlur(img_binary[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = img_binary_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
                                hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

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

    vertices = compute_values(0, 71, 49, 3, 51, 1, 1)

    if vertices is None:
        if blackout:
            return None, img_in
        else:
            return None

    center = _get_polygon_center(vertices)

    if blackout:
        return center, img_in

    return center


def construction_sign_detection(img_in, blackout=False):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    bgr = [0, 127, 255]
    tolerance = 5
    img_binary = color_filter_bgr(img_in, bgr, tolerance)

    def compute_values(blur_sigma, canny_min, canny_max, hough_angle_resolution,
                       hough_threshold,
                       houghP_minLineLength, houghP_maxLineGap):
        img_binary_blur = cv2.GaussianBlur(img_binary[:, :, 2], (5, 5), blur_sigma)

        image_for_canny = img_binary_blur

        edges = cv2.Canny(image_for_canny, canny_min, canny_max)

        lines = cv2.HoughLinesP(edges, 1, hough_angle_resolution * np.pi / 180,
                                hough_threshold, houghP_minLineLength, houghP_maxLineGap)

        return lines

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

    vertices = compute_values(10, 10, 10, 15, 50, 0, 46)

    if vertices is None:
        return None

    center = _get_polygon_center(vertices)

    if blackout:
        return center, img_in

    return center


def do_not_enter_sign_detection(img_in, blackout=False):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img_gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    bgr = [0, 0, 255]
    tolerance = 2
    img_binary = color_filter_bgr(img_in, bgr, tolerance)
    img_binary_gray = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)

    def compute_values(accumulator_resolution, min_dist, param1, param2, min_radius, max_radius):
        radii_range = range(min_radius, max_radius)

        collections = []
        for radius in radii_range:
            circles = cv2.HoughCircles(img_binary_gray,
                                       cv2.HOUGH_GRADIENT,
                                       accumulator_resolution,  # inverse ratio, accumulator resolution
                                       min_dist,  # minDist between circle centers
                                       param1=param1,  # Canny edge detector upper threshold
                                       param2=param2,  # Accumulator value for circle centers
                                       minRadius=radius,
                                       maxRadius=radius)
            if circles is not None:
                if len(circles[0]) == 1:
                    collections.append(circles)

        if collections:
            detections = sum(collections) / len(collections)
            detections = detections.astype(np.uint64)
            return detections
        else:
            return None

    def draw_image_hough(circles):
        output_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if circles is None:
            return output_img
        print(len(circles[0]))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(i)

            output_img = _add_cross_hairs(output_img, (int(i[0]), int(i[1])))

        return output_img.astype(dtype=np.uint8)

    # last_args = display_trackbar_window(
    #     'do_not_enter',
    #     draw_image_hough,
    #     compute_values,
    #     accumulator_resolution=param(3, 1),
    #     min_dist=param(80, 10),
    #     param1=param(100, 50),
    #     param2=param(20, 8),
    #     min_radius=param(100, 10),
    #     max_radius=param(100, 30),
    # )

    circle_coordinates = compute_values(1, 80, 50, 8, 20, 45)

    if circle_coordinates is None:
        return None

    x, y, r = tuple(int(i) for i in circle_coordinates[0, 0, :])

    if blackout:
        cv2.circle(img_in, (x, y), r+2, (0, 0, 0), -1)
        return (x, y), img_in

    return (x, y)


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
    handlers = {
        'no_entry': do_not_enter_sign_detection,
        'stop': stop_sign_detection,
        'construction': construction_sign_detection,
        'warning': warning_sign_detection,
        'traffic_light': traffic_light_detection,
        'yield': yield_sign_detection,
    }

    img_copy = np.copy(img_in)
    signs_present = {}

    for name, handler in handlers.items():
        # import pdb; pdb.set_trace()
        print("**********", name)

        if name == 'traffic_light':
            # import pdb; pdb.set_trace()
            radii_range = range(5, 30, 1)
            center, _, img_copy= handler(img_copy, radii_range, blackout=True)
        else:
            center, img_copy = handler(img_copy, blackout=True)
        if center:
            signs_present[name] = center

    print(signs_present)

    output_img = np.copy(img_in)

    for name, value in signs_present.items():
        output_img = _add_cross_hairs(output_img, value)

    return signs_present


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
