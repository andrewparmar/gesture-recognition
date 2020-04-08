import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import os

import cv2
import numpy as np

from config import actions, backgrounds, frame_sequences, training_sequence

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
            self.images[:, :, -1] = image
            self.last_image = image
        else:
            self.images[:, :, : self.n - 1] = self.images[:, :, 1 : self.n]
            self.images[:, :, -1] = self.last_image
            self.last_image = image

    def get_binary_image(self, mode="frame"):
        """
        binary_image is of type np.float64
        """
        if mode == "median":
            median_image = np.median(self.images[:, :, : self.n - 1], axis=2).astype(
                np.uint8
            )
            diff_image = cv2.absdiff(self.last_image, median_image)
        else:
            diff_image = cv2.absdiff(self.last_image, self.images[:, :, self.n - 1])

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
        if type == "mhi":
            cv2.namedWindow("motion_history_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_history_image", (600, 600))
            cv2.imshow("motion_history_image", self.mhi)
            cv2.waitKey(WAIT_DURATION)
        elif type == "mei":
            cv2.namedWindow("motion_energy_image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("motion_energy_image", (600, 600))
            cv2.imshow("motion_energy_image", self.mei)
            cv2.waitKey(WAIT_DURATION)


class HuMoments:
    def __init__(self, image):
        self._moments = self._calc_moments(image)
        self.values = self._calc_hu_moments()

    def _calc_moments(self, image):
        y, x = np.mgrid[: image.shape[0], : image.shape[1]]

        if image.sum() == 0:
            keys = ["nu11", "nu12", "nu21", "nu02", "nu20", "nu03", "nu30"]
            moments = {k: 0 for k in keys}
            return moments
        else:
            x_mean = (x * image).sum() / image.sum()
            y_mean = (y * image).sum() / image.sum()

        moments = {}

        # regular moments
        moments["m00"] = (x ** 0 * y ** 0 * image).sum()
        moments["m10"] = (x ** 1 * y ** 0 * image).sum()
        moments["m01"] = (x ** 0 * y ** 1 * image).sum()
        moments["m11"] = (x ** 1 * y ** 1 * image).sum()
        moments["m20"] = (x ** 2 * y ** 0 * image).sum()
        moments["m02"] = (x ** 0 * y ** 2 * image).sum()
        moments["m30"] = (x ** 3 * y ** 0 * image).sum()
        moments["m03"] = (x ** 0 * y ** 3 * image).sum()
        moments["m12"] = (x ** 1 * y ** 2 * image).sum()
        moments["m21"] = (x ** 2 * y ** 1 * image).sum()

        # central moments
        moments["mu10"] = ((x - x_mean) ** 1 * (y - y_mean) ** 0 * image).sum()
        moments["mu01"] = ((x - x_mean) ** 0 * (y - y_mean) ** 1 * image).sum()
        moments["mu11"] = ((x - x_mean) ** 1 * (y - y_mean) ** 1 * image).sum()
        moments["mu20"] = ((x - x_mean) ** 2 * (y - y_mean) ** 0 * image).sum()
        moments["mu02"] = ((x - x_mean) ** 0 * (y - y_mean) ** 2 * image).sum()
        moments["mu30"] = ((x - x_mean) ** 3 * (y - y_mean) ** 0 * image).sum()
        moments["mu03"] = ((x - x_mean) ** 0 * (y - y_mean) ** 3 * image).sum()
        moments["mu12"] = ((x - x_mean) ** 1 * (y - y_mean) ** 2 * image).sum()
        moments["mu21"] = ((x - x_mean) ** 2 * (y - y_mean) ** 1 * image).sum()

        # scale invariant moments
        moments["nu11"] = moments["mu11"] / moments["m00"] ** (1 + (1 + 1) / 2)
        moments["nu12"] = moments["mu12"] / moments["m00"] ** (1 + (1 + 2) / 2)
        moments["nu21"] = moments["mu21"] / moments["m00"] ** (1 + (2 + 1) / 2)
        moments["nu02"] = moments["mu02"] / moments["m00"] ** (1 + (0 + 2) / 2)
        moments["nu20"] = moments["mu20"] / moments["m00"] ** (1 + (2 + 0) / 2)
        moments["nu03"] = moments["mu03"] / moments["m00"] ** (1 + (0 + 3) / 2)
        moments["nu30"] = moments["mu30"] / moments["m00"] ** (1 + (3 + 0) / 2)

        return moments

    def _calc_hu_moments(self):
        nu11 = self._moments["nu11"]
        nu20 = self._moments["nu20"]
        nu02 = self._moments["nu02"]
        nu30 = self._moments["nu30"]
        nu03 = self._moments["nu03"]
        nu12 = self._moments["nu12"]
        nu21 = self._moments["nu21"]

        h1 = nu20 + nu02  # noqa
        h2 = (nu20 - nu02) ** 2 + 4 * nu11 ** 2  # noqa
        h3 = (nu30 - 3 * nu12) ** 2 + (3 * nu21 - nu03) ** 2  # noqa
        h4 = (nu30 + nu12) ** 2 + (nu21 + nu03) ** 2  # noqa
        h5 = (nu30 - 3 * nu12) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
        ) + (3 * nu21 - nu03) * (nu21 + nu03) * (
            3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )  # noqa
        h6 = (nu20 - nu02) * ((nu30 + nu12) ** 2 - (nu21 + nu03) ** 2) + 4 * nu11 * (
            nu30 + nu12
        ) * (
            nu21 + nu03
        )  # noqa
        h7 = (3 * nu21 - nu03) * (nu30 + nu12) * (
            (nu30 + nu12) ** 2 - 3 * (nu21 + nu03) ** 2
        ) - (nu30 - 3 * nu12) * (nu21 + nu03) * (
            3 * (nu30 + nu12) ** 2 - (nu21 + nu03) ** 2
        )  # noqa

        # hu_moments = np.nan_to_num(np.array([h1, h2, h3, h4, h5, h6, h7]))
        hu_moments = np.array([h1, h2, h3, h4, h5, h6, h7])

        return hu_moments


class ActionVideo:
    PARAM_MAP = {
        'boxing':       {"label": 1, "theta": 20, "ksize": 2},  # noqa
        'handclapping': {"label": 2, "theta": 20, "ksize": 2},  # noqa
        'handwaving':   {"label": 3, "theta": 20, "ksize": 2},  # noqa
        "jogging":      {"label": 4, "theta": 20, "ksize": 2},  # noqa
        "running":      {"label": 5, "theta": 20, "ksize": 2},  # noqa
        "walking":      {"label": 6, "theta": 50, "ksize": 2},  # noqa
    }
    NUM_HU = 7

    def __init__(self, num, action, background):
        self.key_name = f"person{num:02d}_{action}_{background}"
        self.filename = f"{self.key_name}_uncomp.avi"
        self.action = action
        self.frame_ranges = frame_sequences[self.key_name]
        self.video_frame_array = None

        self._video_to_image_array()

    def _video_to_image_array(self):
        input_video_path = os.path.join(VID_DIR, self.filename)

        self.total_video_frames = self._get_video_frame_count(input_video_path)

        input_image_gen = self._gray_frame_generator(input_video_path)
        input_image = input_image_gen.__next__()

        self.video_frame_array = np.zeros(
            (input_image.shape[0], input_image.shape[1], self.total_video_frames),
            dtype=np.uint8,
        )

        frame_num = 0

        while input_image is not None:
            self.video_frame_array[:, :, frame_num] = input_image

            input_image = input_image_gen.__next__()

            frame_num += 1

    def _get_video_frame_count(self, file_path):
        video = cv2.VideoCapture(file_path)

        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video.release()

        return total

    @staticmethod
    def _gray_frame_generator(file_path):
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

    def analyze_frames(self):
        n = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = 10

        frame_count_total = 0
        for frame_range in self.frame_ranges:
            frame_count_total += (frame_range[1] - frame_range[0] + 1)
        print(f'Frame range count total: {frame_count_total}')
        print(f'Total frames: {self.total_video_frames}')

        self.frame_features = np.zeros((frame_count_total, self.NUM_HU))
        self.frame_labels = np.zeros(frame_count_total)

        counter = 0

        for frame_range in self.frame_ranges:

            binary_motion = BinaryMotion(n, theta)
            temporal_template = TemporalTemplate(tau)

            start = frame_range[0] - 1
            end = frame_range[1]
            print(f"{start}:{end}")

            for i in range(start, end):
                input_image_t = self.video_frame_array[:, :, i]

                binary_motion.update(input_image_t)
                temporal_template.update(binary_motion.get_binary_image())

                binary_motion.view()
                temporal_template.view(type="mhi")

                hu_moments = HuMoments(temporal_template.mei)
                # print(f"HuMoments: {hu_moments.values}")
                self.frame_features[counter] = hu_moments.values
                if (self.frame_features[counter].sum() != 0):
                    self.frame_labels[counter] = self.PARAM_MAP[self.action]['label']
                counter += 1

    def analyze_frames_2(self):
        """
        Uses vstack to accumulate frame arrays.
        """
        n = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = 10

        self.frame_features = np.zeros((1, self.NUM_HU))
        self.frame_labels = np.array([0])

        counter = 0

        for frame_range in self.frame_ranges:

            binary_motion = BinaryMotion(n, theta)
            temporal_template = TemporalTemplate(tau)

            start = frame_range[0] - 1
            end = frame_range[1]
            # print(f"{start}:{end}")

            for i in range(start, end):
                input_image_t = self.video_frame_array[:, :, i]

                binary_motion.update(input_image_t)
                temporal_template.update(binary_motion.get_binary_image())
                # cv2.imwrite(f"output_images/{i}_binary.jpg", binary_motion.get_binary_image() * 255)

                # cv2.imwrite(f"output_images/{i}_mhi.jpg", temporal_template.mhi)

                # binary_motion.view()
                # temporal_template.view(type="mhi")

                # if counter == 141:
                #     cv2.imwrite(f"output_images/{counter}_binary.jpg", binary_motion.get_binary_image() * 255)
                #     cv2.imwrite(f"output_images/{counter}_mhi.jpg", temporal_template.mhi)

                hu_moments = HuMoments(temporal_template.mei)
                # print(hu_moments.values)
                label = np.array([self.PARAM_MAP[self.action]['label']])

                if np.any(hu_moments.values):
                    self.frame_features = np.vstack((self.frame_features, hu_moments.values))
                    self.frame_labels = np.vstack((self.frame_labels, label))

                counter += 1

        self.frame_features = self.frame_features[1:, :]
        self.frame_labels = self.frame_labels[1:]

    def analyze_frames_all(self):
        n = 2
        theta = self.PARAM_MAP[self.action]["theta"]
        tau = 10

        self.frame_features = np.zeros((self.total_video_frames, self.NUM_HU))
        self.frame_labels = np.zeros(self.total_video_frames)

        for frame_range in self.frame_ranges:

            binary_motion = BinaryMotion(n, theta)
            temporal_template = TemporalTemplate(tau)

            start = frame_range[0] - 1
            end = frame_range[1]

            for i in range(start, end):
                input_image_t = self.video_frame_array[:, :, i]

                binary_motion.update(input_image_t)
                temporal_template.update(binary_motion.get_binary_image())

                # binary_motion.view()
                # temporal_template.view(type="mhi")
                hu_moments = HuMoments(temporal_template.mhi)
                print(f"HuMoments: {hu_moments.values}")
                self.frame_features[i] = hu_moments.values


def generate_training_data():
    """
    Example:
    "person01_walking_d1": [(1, 75), (152, 225), (325, 400), (480, 555)],
    """

    Xtrain = np.zeros((1, 7))
    ytrain = np.zeros((1, 1))

    for person_num in training_sequence[:1]:
        for action in actions:
            for background in backgrounds[:1]:

                # Manual override.
                # person_num = 1
                # action = 'walking'
                # background = 'd1'
                # import pdb; pdb.set_trace()
                # print(person_num, action, background)

                action_video = ActionVideo(person_num, action, background)
                print(action_video.key_name)
                action_video.analyze_frames_2()
                # import pdb; pdb.set_trace()

                start, end = action_video.frame_ranges[0]

                features = np.abs(np.log(np.abs(action_video.frame_features)))
                features[features == -np.inf] = 0
                features[features == np.nan] = 0
                # features = np.nan_to_num(features)

                # features = np.abs(np.log(np.abs(action_video.frame_features[start:end, :])))
                # print('*'*80)
                # features[features == -np.inf] = 0
                # features = np.nan_to_num(features)
                # print(features[start:end, :])

                ##########################################################################
                # fig, ax = plt.subplots()
                # ax.plot(features)
                # # ax.plot(np.log(action_video.frame_features[:, :]))
                #
                # labels = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
                #
                # ax.legend(labels,
                #           bbox_to_anchor=(1, 1),
                #           # loc='best'
                #           )
                # ax.set(
                #     xlabel='Frame number',
                #     ylabel='Hu Value',
                #     title=f'{action_video.key_name}'
                # )
                # plt.show()
                ##########################################################################

                Xtrain = np.vstack((Xtrain, action_video.frame_features))
                # import pdb; pdb.set_trace()
                ytrain = np.hstack((ytrain, action_video.frame_labels.T))

    return Xtrain[1:], ytrain[1:]


# def plot_features(features, labels):
#     fig, ax = plt.subplots()
#     ax.plot(features)
#     # ax.plot(np.log(action_video.frame_features[:, :]))
#
#     labels = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
#
#     ax.legend(labels,
#               bbox_to_anchor=(1, 1),
#               # loc='best'
#               )
#     ax.set(
#         xlabel='Frame number',
#         ylabel='Hu Value',
#         title=f'{action_video.key_name}'
#     )
#     plt.show()
