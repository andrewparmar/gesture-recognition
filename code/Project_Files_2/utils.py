import os

import cv2
import numpy as np

VID_DIR = "sample_dataset"


class BinaryMotion:

    def __init__(self, len_img_history, threshold):
        self.n = len_img_history
        self.theta = threshold

    def update(self, image):
        if not hasattr(self, 'last_image'):
            h, w = image.shape
            self.images = np.zeros((h, w, self.n), dtype=np.uint8)
            self.last_image = image
            self.binary_image = np.zeros((h, w))
        else:
            self.images[:, :, :self.n - 1] = self.images[:, :, 1:self.n]
            self.images[:, :, -1] = self.last_image
            self.last_image = image

    def get_binary_image(self):
        diff_image = cv2.absdiff(self.last_image, np.median(self.images, axis=2).astype(np.uint8))

        self.binary_image = np.zeros(diff_image.shape)

        idx = diff_image >= self.theta
        self.binary_image[idx] = 1

        self._cleanup()

        return self.binary_image

    def _cleanup(self):
        kernel = np.ones((5, 5), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)


def get_binary_image(image1, image2, threshold):
    # TODO: median filtering. See 8D-L1 lecture

    diff_image = cv2.absdiff(image1, image2)

    binary_image = np.zeros(diff_image.shape)

    idx = diff_image >= threshold

    binary_image[idx] = 1

    binary_image = cleanup_image(binary_image)

    return binary_image


def make_motion_history_image_recursive(binary_sequence):
    # Second try: Recursive reduction
    h, w, n = binary_sequence.shape

    mhi = np.zeros((h, w))

    tau = 255

    if n == 1:
        idx = binary_sequence[:, :, 0] == 1
        mhi[idx] = tau
        mhi[~idx] = 0
        return mhi
    else:
        idx = binary_sequence[:, :, 0] == 1
        mhi[idx] = tau
        mhi[~idx] = np.maximum(make_motion_history_image_recursive(binary_sequence[:, :, 1:]) - 30, 0)[~idx]
        return mhi


def make_motion_history_image(binary_sequence):
    # First try: Reduce images by set weight
    # import pdb; pdb.set_trace()
    h, w, n = binary_sequence.shape

    mhi = np.zeros((h, w))

    tau = 255 / n

    for i in range(n):
        print(f"This is {i}th frame. Tau: {255 - tau*i}")
        idx = binary_sequence[:, :, i] == 1
        mhi[idx] = tau * i

    return mhi


def cleanup_image(image):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return cleaned_image


def video_to_image_array(filename, fps):

    input_video_path = os.path.join(VID_DIR, filename)
    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image_t = input_image_gen.__next__()

    h, w = input_image_t.shape
    n = 50  # 50 is best so far
    theta = 40  # 50 is best so far
    q = 10

    frame_num = 0

    binary_motion = BinaryMotion(n, theta)

    last_q_binary_images = np.zeros((h, w, q), dtype=np.uint8)

    while input_image_t is not None:
        binary_motion.update(input_image_t)

        if frame_num % 10 == 0:
            print("Processing fame {}".format(frame_num))

        # if frame_num == 200:
        #     import pdb; pdb.set_trace()

        binary_image = binary_motion.get_binary_image()

        last_q_binary_images[:, :, : q - 1] = last_q_binary_images[:, :, 1:q]
        last_q_binary_images[:, :, -1] = binary_image

        # motion_history_image = make_motion_history_image(np.flip(last_q_binary_images, 2))
        motion_history_image = make_motion_history_image(last_q_binary_images)

        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", binary_image)

        cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("motion_history_image", (600, 600))
        cv2.imshow("motion_history_image", motion_history_image.astype(np.uint8))

        cv2.waitKey(10)

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
