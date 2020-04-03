import os

import cv2
import numpy as np

VID_DIR = "sample_dataset"
WAIT_DURATION = 1


class BinaryMotion:
    def __init__(self, len_img_history, threshold):
        self.n = len_img_history
        self.theta = threshold
        self.images = None
        self.last_image = None
        self.binary_image = None

    def update(self, image):
        if self.images is None:
            h, w = image.shape
            self.images = np.zeros((h, w, self.n), dtype=np.uint8)
            self.last_image = image
        else:
            self.images[:, :, : self.n - 1] = self.images[:, :, 1 : self.n]
            self.images[:, :, -1] = self.last_image
            self.last_image = image

    def get_binary_image(self):
        """
        binary_image is of type np.float64
        """
        diff_image = cv2.absdiff(
            self.last_image, np.median(self.images, axis=2).astype(np.uint8)
        )

        self.binary_image = np.zeros(diff_image.shape)
        self.binary_image[diff_image >= self.theta] = 1
        self._cleanup()

        return self.binary_image

    def _cleanup(self):
        kernel = np.ones((5, 5), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def view(self):
        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", self.binary_image)
        cv2.waitKey(WAIT_DURATION)


class MotionHistoryImage:
    def __init__(self, len_img_history):
        self.n = len_img_history
        self.tau = 255 / self.n
        self.motion_images = None
        self.mhi = None
        # Note: self.mei could just be mhi as float right?

    def update(self, motion_img):
        if self.motion_images is None:
            h, w = motion_img.shape
            self.motion_images = np.zeros((h, w, self.n), dtype=np.uint8)
        else:
            self.motion_images[:, :, : self.n - 1] = self.motion_images[
                :, :, 1 : self.n
            ]  # noqa

        self.motion_images[:, :, -1] = motion_img

        self._make_motion_history_image()

    def _make_motion_history_image(self):
        h, w, n = self.motion_images.shape

        self.mhi = np.zeros((h, w))

        for i in range(n):
            idx = self.motion_images[:, :, i] == 1
            self.mhi[idx] = self.tau * i

        self.mhi = self.mhi.astype(np.uint8)

    def view(self):
        cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_history_image", (600, 600))
        cv2.imshow("motion_history_image", self.mhi)
        cv2.waitKey(WAIT_DURATION)


# def moment(image):
#     h, w = image.shape
#
#     m00 = 0
#     m10 = 0
#     m01 = 0
#
#     for y in range(h):
#
#         for x in range(w):
#             m00 += image[y, x]
#
#             m10 += x ** 1 * y ** 0 * image[y, x]
#
#             m01 += x ** 0 * y ** 1 * image[y, x]
#
#     x_mean = m10 / m00
#     y_mean = m01 / m00
#
#     mu00 = 0
#     mu10 = 0
#     mu01 = 0
#     mu11 = 0
#     mu20 = 0
#     mu02 = 0
#     mu30 = 0
#     mu03 = 0
#     mu12 = 0
#     mu21 = 0
#
#     for y in range(h):
#
#         for x in range(w):
#             mu00 += image[y, x]
#
#             mu10 += (x - x_mean) ** 1 * (y - y_mean) ** 0 * image[y, x]
#
#             mu01 += (x - x_mean) ** 0 * (y - y_mean) ** 1 * image[y, x]
#
#             mu11 += (x - x_mean) ** 1 * (y - y_mean) ** 1 * image[y, x]
#
#             mu20 += (x - x_mean) ** 2 * (y - y_mean) ** 0 * image[y, x]
#
#             mu02 += (x - x_mean) ** 0 * (y - y_mean) ** 2 * image[y, x]
#
#             mu30 += (x - x_mean) ** 3 * (y - y_mean) ** 0 * image[y, x]
#
#             mu03 += (x - x_mean) ** 0 * (y - y_mean) ** 3 * image[y, x]
#
#             mu12 += (x - x_mean) ** 1 * (y - y_mean) ** 2 * image[y, x]
#
#             mu21 += (x - x_mean) ** 2 * (y - y_mean) ** 1 * image[y, x]
#
#     print(
#         mu00, mu10, mu01, mu11, mu20, mu02, mu30, mu03, mu12, mu21,
#     )
#
#     print(image.sum())

def moments(image):
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]

    moments = {}
    moments['mean_x'] = (x * image).sum() / image.sum()
    moments['mean_y'] = (y * image).sum() / image.sum()

    # raw or spatial moments
    moments['m00'] = (image).sum()
    moments['m10'] = (x ** 1 * y ** 0 * image).sum()
    moments['m01'] = (x ** 0 * y ** 1 * image).sum()
    moments['m11'] = (x ** 1 * y ** 1 * image).sum()
    moments['m20'] = (x ** 2 * y ** 0 * image).sum()
    moments['m02'] = (x ** 0 * y ** 2 * image).sum()
    moments['m30'] = (x ** 3 * y ** 0 * image).sum()
    moments['m03'] = (x ** 0 * y ** 3 * image).sum()
    moments['m12'] = (x ** 1 * y ** 2 * image).sum()
    moments['m21'] = (x ** 2 * y ** 1 * image).sum()

    # central moments
    x_mean = moments['mean_x']
    y_mean = moments['mean_y']
    moments['mu10'] = ((x - x_mean) ** 1 * (y - y_mean) ** 0 * image).sum()
    moments['mu01'] = ((x - x_mean) ** 0 * (y - y_mean) ** 1 * image).sum()
    moments['mu11'] = ((x - x_mean) ** 1 * (y - y_mean) ** 1 * image).sum()
    moments['mu20'] = ((x - x_mean) ** 2 * (y - y_mean) ** 0 * image).sum()
    moments['mu02'] = ((x - x_mean) ** 0 * (y - y_mean) ** 2 * image).sum()
    moments['mu30'] = ((x - x_mean) ** 3 * (y - y_mean) ** 0 * image).sum()
    moments['mu03'] = ((x - x_mean) ** 0 * (y - y_mean) ** 3 * image).sum()
    moments['mu12'] = ((x - x_mean) ** 1 * (y - y_mean) ** 2 * image).sum()
    moments['mu21'] = ((x - x_mean) ** 2 * (y - y_mean) ** 1 * image).sum()

    # central standardized or normalized or scale invariant moments
    moments['nu11'] = moments['mu11'] / moments['m00'] ** (1 + (1 + 1) / 2)
    moments['nu12'] = moments['mu12'] / moments['m00'] ** (1 + (1 + 2) / 2)
    moments['nu21'] = moments['mu21'] / moments['m00'] ** (1 + (2 + 1) / 2)
    moments['nu02'] = moments['mu02'] / moments['m00'] ** (1 + (0 + 2) / 2)
    moments['nu20'] = moments['mu20'] / moments['m00'] ** (1 + (2 + 0) / 2)
    moments['nu03'] = moments['mu03'] / moments['m00'] ** (1 + (0 + 3) / 2)
    moments['nu30'] = moments['mu30'] / moments['m00'] ** (1 + (3 + 0) / 2)

    return moments

def hu_moments(moments):
    # nu10 = moments['nu10']
    # nu01 = moments['nu01']
    nu11 = moments['nu11']
    nu20 = moments['nu20']
    nu02 = moments['nu02']
    nu30 = moments['nu30']
    nu03 = moments['nu03']
    nu12 = moments['nu12']
    nu21 = moments['nu21']

    h1 = nu20 + nu02  # noqa
    h2 = (nu20 - nu02)**2 + 4*nu11**2  # noqa
    h3 = (nu30 - 3*nu12)**2 + (3*nu21 - nu03)**2  # noqa
    h4 = (nu30 + nu12)**2 + (nu21 + nu03)**2  # noqa
    h5 = (nu30 - 3*nu12)*(nu30 + nu12) * ((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) + (3*nu21 - nu03) * (nu21 + nu03) * (3*(nu30 + nu12)**2 - (nu21 + nu03)**2)  # noqa
    h6 = (nu20 - nu02)*((nu30 + nu12)**2 - (nu21 + nu03)**2) + 4 * nu11 * (nu30 + nu12) *(nu21 + nu03)  # noqa
    h7 = (3 * nu21 - nu03) * (nu30 + nu12) * ((nu30 + nu12)**2 - 3*(nu21 + nu03)**2) - (nu30 - 3*nu12)*(nu21 + nu03)*(3*(nu30 + nu12)**2 - (nu21 + nu03)**2)  # noqa

    hu_moments = [h1, h2, h3, h4, h5, h6, h7]

    return hu_moments


def video_to_image_array(filename, fps):
    input_video_path = os.path.join(VID_DIR, filename)
    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image_t = input_image_gen.__next__()

    n = 50  # 50 is best so far
    theta = 40  # 50 is best so far
    q = 10

    frame_num = 0

    binary_motion = BinaryMotion(n, theta)
    motion_history_image = MotionHistoryImage(q)

    while input_image_t is not None:
        if frame_num % 20 == 0:
            print("Processing fame {}".format(frame_num))
        if frame_num == 200:
            cv2.imwrite(
                "mhi_frame_200_person01_walking_d1.png", motion_history_image.mhi
            )

        binary_motion.update(input_image_t)
        motion_history_image.update(binary_motion.get_binary_image())

        binary_motion.view()
        motion_history_image.view()

        input_image_t = input_image_gen.__next__()

        frame_num += 1


def video_gray_frame_generator(file_path):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        file_path (string): Relative file path.

    Returns:
        None.
    """
    video = cv2.VideoCapture(file_path)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray_image
            # yield gray_image
        else:
            break

    video.release()
    yield None
