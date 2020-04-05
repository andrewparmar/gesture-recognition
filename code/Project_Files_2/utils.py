import os
import sys

import cv2
import numpy as np

from config import actions, backgrounds, training_sequence, frame_sequences

VID_DIR = "sample_dataset"
WAIT_DURATION = 1


class BinaryMotion:
    def __init__(self, len_img_history, threshold):
        self.n = len_img_history
        self.theta = threshold
        self.images = None
        self.last_image = None
        self.binary_image = None
        self.ksize = 2

    def update(self, image):
        if self.images is None:
            h, w = image.shape
            self.images = np.zeros((h, w, self.n), dtype=np.uint8)
            self.last_image = image
        else:
            self.images[:, :, : self.n - 1] = self.images[:, :, 1 : self.n]
            self.images[:, :, -1] = self.last_image
            self.last_image = image

    def get_binary_image(self, mode='frame'):
        """
        binary_image is of type np.float64
        """
        if mode == 'median':
            median_image = np.median(self.images[:,:,:self.n-1], axis=2).astype(np.uint8)
            diff_image = cv2.absdiff(self.last_image, median_image)
        else:
            diff_image = cv2.absdiff(self.last_image, self.images[:, :, self.n-1])

        self.binary_image = np.zeros(diff_image.shape)
        self.binary_image[diff_image >= self.theta] = 1
        self._cleanup()

        return self.binary_image

    def _cleanup(self):
        kernel = np.ones((self.ksize, self.ksize), np.uint8)
        self.binary_image = cv2.morphologyEx(self.binary_image, cv2.MORPH_OPEN, kernel)

    def view(self):
        cv2.namedWindow("binary_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binary_image", (600, 600))
        cv2.imshow("binary_image", self.binary_image)
        cv2.waitKey(WAIT_DURATION)


class TemporalTemplate:
    def __init__(self, len_img_history):
        self.n = len_img_history
        self.tau = 255 / self.n
        self.motion_images = None
        self.mhi = None
        self.mei = None
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
        self._make_motion_energy_image()

    def _make_motion_history_image(self):
        h, w, n = self.motion_images.shape

        self.mhi = np.zeros((h, w))

        for i in range(n):
            idx = self.motion_images[:, :, i] == 1
            self.mhi[idx] = self.tau * i

        self.mhi = self.mhi.astype(np.uint8)

    def _make_motion_energy_image(self):
        h, w, n = self.motion_images.shape

        self.mei = np.zeros((h, w))

        for i in range(n):
            idx = self.motion_images[:, :, i] == 1
            self.mei[idx] = 255

        self.mei = self.mei.astype(np.uint8)

    def view(self, type):
        if type == 'mhi':
            cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_history_image", (600, 600))
            cv2.imshow("motion_history_image", self.mhi)
            cv2.waitKey(WAIT_DURATION)
        elif type == 'mei':
            cv2.namedWindow("motion_energy_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_energy_image", (600, 600))
            cv2.imshow("motion_energy_image", self.mei)
            cv2.waitKey(WAIT_DURATION)


def moments(image):
    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    x_mean = (x * image).sum() / image.sum()
    y_mean = (y * image).sum() / image.sum()

    moments = {}

    # regular moments
    moments['m00'] = (x ** 0 * y ** 0 * image).sum()
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
    moments['mu10'] = ((x - x_mean) ** 1 * (y - y_mean) ** 0 * image).sum()
    moments['mu01'] = ((x - x_mean) ** 0 * (y - y_mean) ** 1 * image).sum()
    moments['mu11'] = ((x - x_mean) ** 1 * (y - y_mean) ** 1 * image).sum()
    moments['mu20'] = ((x - x_mean) ** 2 * (y - y_mean) ** 0 * image).sum()
    moments['mu02'] = ((x - x_mean) ** 0 * (y - y_mean) ** 2 * image).sum()
    moments['mu30'] = ((x - x_mean) ** 3 * (y - y_mean) ** 0 * image).sum()
    moments['mu03'] = ((x - x_mean) ** 0 * (y - y_mean) ** 3 * image).sum()
    moments['mu12'] = ((x - x_mean) ** 1 * (y - y_mean) ** 2 * image).sum()
    moments['mu21'] = ((x - x_mean) ** 2 * (y - y_mean) ** 1 * image).sum()

    # scale invariant moments
    moments['nu11'] = moments['mu11'] / moments['m00'] ** (1 + (1 + 1) / 2)
    moments['nu12'] = moments['mu12'] / moments['m00'] ** (1 + (1 + 2) / 2)
    moments['nu21'] = moments['mu21'] / moments['m00'] ** (1 + (2 + 1) / 2)
    moments['nu02'] = moments['mu02'] / moments['m00'] ** (1 + (0 + 2) / 2)
    moments['nu20'] = moments['mu20'] / moments['m00'] ** (1 + (2 + 0) / 2)
    moments['nu03'] = moments['mu03'] / moments['m00'] ** (1 + (0 + 3) / 2)
    moments['nu30'] = moments['mu30'] / moments['m00'] ** (1 + (3 + 0) / 2)

    return moments


def hu_moments(moments):
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

def get_hu_moments(image):
    moments_ = moments(image)
    return hu_moments(moments_)


def video_frame_sequence_analyzer(filename, fps):
    input_video_path = os.path.join(VID_DIR, filename)
    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image_t = input_image_gen.__next__()

    n = 2  # 50 is best so far
    theta = 20  # for frame differencing. (50 is best so far with median)
    q = 10

    frame_num = 0

    binary_motion = BinaryMotion(n, theta)
    temporal_template = TemporalTemplate(q)

    while input_image_t is not None:
        if frame_num % 20 == 0:
            print("Processing fame {}".format(frame_num))
        # if frame_num == 200:
        #     cv2.imwrite("mhi_frame_200_person01_walking_d1.png", temporal_template.mhi)

        binary_motion.update(input_image_t)
        temporal_template.update(binary_motion.get_binary_image())

        binary_motion.view()
        temporal_template.view(type='mhi')
        # temporal_template.view(type='mei')

        # print(f'Frame:{frame_num}; HuMoment: {get_hu_moments(temporal_template.mhi)}')

        input_image_t = input_image_gen.__next__()

        frame_num += 1

def video_frame_array_analyzer(video_frame_array, frame_ranges):
    n = 2
    theta = 20  # for frame differencing. (50 is best so far with median)
    q = 10

    for frame_range in frame_ranges:

        binary_motion = BinaryMotion(n, theta)
        temporal_template = TemporalTemplate(q)

        start = frame_range[0] - 1
        end = frame_range[1]

        for i in range(start, end):

            input_image_t = video_frame_array[:,:,i]

            binary_motion.update(input_image_t)
            temporal_template.update(binary_motion.get_binary_image())

            binary_motion.view()
            temporal_template.view(type='mhi')


def video_to_image_array(filename):
    input_video_path = os.path.join(VID_DIR, filename)

    total_video_frames = get_video_frame_count(input_video_path)

    input_image_gen = video_gray_frame_generator(input_video_path)
    input_image = input_image_gen.__next__()

    video_image_array = np.zeros(
        (input_image.shape[0], input_image.shape[1], total_video_frames), dtype=np.uint8)

    frame_num = 0

    while input_image is not None:

        video_image_array[:,:,frame_num] = input_image

        input_image = input_image_gen.__next__()

        frame_num += 1

    return video_image_array


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
        else:
            break

    video.release()
    yield None


def get_video_frame_count(file_path):
    video = cv2.VideoCapture(file_path)

    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()

    return total


def dataset_loop():
    """
    "person01_walking_d1": [(1, 75), (152, 225), (325, 400), (480, 555)],
    """

    for num in training_sequence[:1]:
        for action in actions:
            for background in backgrounds:
                key_name = f'person{num:02d}_{action}_{background}'
                filename = f'{key_name}_uncomp.avi'

                # convert video to image array
                video_frame_array = video_to_image_array(filename)
                print(f'Final video array shape {video_frame_array.shape}')

                # from each frame sequence, get huMoments
                video_frame_array_analyzer(video_frame_array, frame_sequences[key_name])

                # create training array.


action_theta = {
    'walking': {'theta': 20, 'ksize':2},
    'jogging': {'theta': 20, 'ksize':2},

}