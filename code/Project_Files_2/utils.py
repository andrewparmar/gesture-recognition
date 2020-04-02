import os

import cv2
import numpy as np

VID_DIR = "sample_dataset"


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
        cv2.waitKey(20)


class MotionHistoryImage:
    def __init__(self, len_img_history):
        self.n = len_img_history
        self.tau = 255 / self.n
        self.motion_images = None
        self.mhi = None

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
        cv2.waitKey(20)


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
        # if frame_num % 10 == 0:
        #     print("Processing fame {}".format(frame_num))
        # if frame_num == 200:
        #     import pdb; pdb.set_trace()

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
